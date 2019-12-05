#server file which handels all the requests that come in

from numpy import * 
import numpy as np
from scipy import optimize  #used to minimize the cost function
from flask import Flask, render_template, request, redirect
from flask_mysqldb import MySQL
import yaml
#render_template is used to render HTML file

app=Flask(__name__)
#configure db
db=yaml.load(open('db.yaml'))
app.config['MYSQL_HOST']=db['mysql_host']
app.config['MYSQL_USER']=db['mysql_user']
app.config['MYSQL_PASSWORD']=db['mysql_password']
app.config['MYSQL_DB']=db['mysql_db']

mysql=MySQL(app)

def normalize_rat(ratings,did_rate):
	num_places=ratings.shape[0]
	num_users=ratings.shape[1]

	ratings_mean=zeros(shape=(num_places,1))
	ratings_norm=zeros(shape=ratings.shape)

	for i in range(num_places):
		c=0
		place=[]
		for j in range(num_users):
			if did_rate[i][j]==1:
				ratings_mean[i]+=ratings[i][j]
				c+=1
		ratings_mean[i]/=c
		if c==0:
			ratings_mean[i]=0
		for j in range(num_users):
			if did_rate[i][j]==1:
				ratings_norm[i][j]=ratings[i][j]-ratings_mean[i]
	return ratings_norm, ratings_mean

def normalize(place_features):
	num_places=place_features.shape[0]

	ratings_mean=zeros(shape=(num_places,1))
	ratings_norm=zeros(shape=place_features.shape)

	for i in range(num_places):
		ratings_mean[i]=mean(place_features[i])
		ratings_norm[i]=place_features[i]-ratings_mean[i]

	return ratings_norm,ratings_mean

def unroll_params(X_and_theta, num_users, num_places, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_places, num_places)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_places * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_places)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_places * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_places, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_places, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def calculate_cost(X_and_theta,ratings,did_rate,num_users,num_places,num_features,reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_places, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	#we are finding the errors in this
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	#regularize to overcome overfitting
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization

@app.route('/recommender')
def recommender():
	cur=mysql.connection.cursor()
	num_places=48
	reg_param=30  #regularisation function

	#contains the ratings given by the users to different places
	cur.execute("SELECT * FROM ratings")
	rate=cur.fetchall()
	num=len(rate)
	ratings=[[0 for i in range(num)] for j in range(48)]

	#transpose the matrix
	for i in range(num):
		for j in range(48):
			ratings[j][i]=rate[i][j]

	num_users=num

	#did_rate checks if a user has rated a places
	did_rate=[[0 for i in range(num_users)] for j in range(num_places)]
	for i in range(num_places):
		for j in range(num_users):
			if ratings[i][j]!=0:
				did_rate[i][j]=1

	ratings=np.array(ratings)
	#print(did_rate)
	#print()
	#return str(ratings)
	#Normalize our data
	#Normalization makes the average of the data as a 0
	ratings,ratings_mean = normalize_rat(ratings,did_rate)
	#return str(ratings_mean)
	#print(ratings)

	#update the number of users
	num_users=ratings.shape[1]
	num_features=13

	#how much of each feature is present
	cur.execute("SELECT * FROM place_features")
	place_features=cur.fetchall()

	place_features=np.array(place_features)
	place_features,place_mean=normalize(place_features)
	#print('place_features')
	#print(place_features)
	#print()
	#return str(place_features)
	
	#what kind of place a user would prefer
	cur.execute("SELECT * FROM user_prefs")
	user_prefs=cur.fetchall()
	user_prefs=np.array(user_prefs)
	user_prefs,user_mean=normalize(user_prefs)
	#print("user_prefs")
	#print(user_prefs)
	#return str(user_prefs)

	#X=place features and theta=user pref y=X*theta
	initial_X_and_theta=r_[place_features.T.flatten(),user_prefs.T.flatten()]

	#return str(initial_X_and_theta)
	#we are going to use gradient descent
	#performing gradient descent
	minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, args=(ratings, did_rate, num_users, num_places, num_features, reg_param),maxiter=100, disp=True, full_output=True ) 
	#return str(minimized_cost_and_optimal_params)
	cost, optimal_place_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]
	#return str(optimal_place_features_and_user_prefs)
	place_features, user_prefs = unroll_params(optimal_place_features_and_user_prefs, num_users, num_places, num_features)
	#print(place_features)
	#return str(place_features)

	all_predictions = place_features.dot( user_prefs.T )
	#return str(ratings_mean)
	
	predictions_for_user=all_predictions[:,0:1]+ratings_mean
	#print("Final ratings I would give to the Places:")
	#return str(all_predictions)
	final_output=[]
	for i in range(num_places):
		final_output.append([predictions_for_user[i],i+1])
	final_output.sort(reverse=True)

	#return str(final_output[1][0])
	cur.execute("SELECT * FROM place_id")
	place_id=cur.fetchall()

	cur.execute("DROP TABLE results")
	cur.execute("CREATE TABLE results (place VARCHAR(32))")
	#return str(final_output)
	for i in range(10):
		cur.execute("SELECT place from place_id WHERE id ="+str(final_output[i][1]))
		#return str(final_output[i][1])
		ans=cur.fetchone()
		#return str(ans)
		m=str(ans[0])
		#return str(m)
		cur.execute("INSERT INTO results (place) VALUES(%s)",[str(m)])
		#return str(pl)
		#print(final_output[i][0]," ",final_output[i][1])
	#print(predictions_for_user)
	#print(len(predictions_for_user))

	cur.execute("SELECT * FROM results")
	userDetails=cur.fetchall()
	mysql.connection.commit()
	cur.close()
	return render_template('users.html',userDetails=userDetails)


@app.route('/ratings',methods=['GET','POST'])
def ratings():
	if request.method=='POST':
		userDetails=request.form
		r1=userDetails['n1']
		r2=userDetails['n2']
		r3=userDetails['n3']
		r4=userDetails['n4']
		r5=userDetails['n5']
		r6=userDetails['n6']
		r7=userDetails['n7']
		r8=userDetails['n8']
		r9=userDetails['n9']
		r10=userDetails['n10']
		r11=userDetails['n11']
		r12=userDetails['n12']
		r13=userDetails['n13']
		r14=userDetails['n14']
		r15=userDetails['n15']
		r16=userDetails['n16']
		r17=userDetails['n17']
		r18=userDetails['n18']
		r19=userDetails['n19']
		r20=userDetails['n20']
		r21=userDetails['n21']
		r22=userDetails['n22']
		r23=userDetails['n23']
		r24=userDetails['n24']
		r25=userDetails['n25']
		r26=userDetails['n26']
		r27=userDetails['n27']
		r28=userDetails['n28']
		r29=userDetails['n29']
		r30=userDetails['n30']
		r31=userDetails['n31']
		r32=userDetails['n32']
		r33=userDetails['n33']
		r34=userDetails['n34']
		r35=userDetails['n35']
		r36=userDetails['n36']
		r37=userDetails['n37']
		r38=userDetails['n38']
		r39=userDetails['n39']
		r40=userDetails['n40']
		r41=userDetails['n41']
		r42=userDetails['n42']
		r43=userDetails['n43']
		r44=userDetails['n44']
		r45=userDetails['n45']
		r46=userDetails['n46']
		r47=userDetails['n47']
		r48=userDetails['n48']
		cur=mysql.connection.cursor()
		cur.execute("INSERT INTO ratings (n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29,n30,n31,n32,n33,n34,n35,n36,n37,n38,n39,n40,n41,n42,n43,n44,n45,n46,n47,n48) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,r32,r33,r34,r35,r36,r37,r38,r39,r40,r41,r42,r43,r44,r45,r46,r47,r48))
		mysql.connection.commit()
		cur.close()
		return redirect('/recommender')
		#return 'success'
	return render_template('rat.html')


@app.route('/userprefs',methods=['GET','POST'])
def userprefs():
	if request.method=='POST':
		#Fetch the form data
		userDetails=request.form
		p1=userDetails['n1']
		p2=userDetails['n2']
		p3=userDetails['n3']
		p4=userDetails['n4']
		p5=userDetails['n5']
		p6=userDetails['n6']
		p7=userDetails['n7']
		p8=userDetails['n8']
		p9=userDetails['n9']
		p10=userDetails['n10']
		p11=userDetails['n11']
		p12=userDetails['n12']
		p13=userDetails['n13']
		cur=mysql.connection.cursor()
		cur.execute("INSERT INTO user_prefs (k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13))
		mysql.connection.commit()
		cur.close()
		return redirect('/ratings')
		#return 'success'
	return render_template('keyword.html')

@app.route('/',methods=['GET','POST'])
def home():
	if request.method=='POST':
		return redirect('/userprefs')
	return render_template('website.html')

if __name__=='__main__':
	app.run(debug=True)