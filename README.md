first install all depencency from requiremnts.txt and hit beloww command in cmd

python fashion.py --precision full --no-half > image1.txt &

you can play with below variables

color = "red"
current_trend = "oversize"
gender = "male"
age = "24"
cloths = "shirt"
inf_step = 10 # is number of inference step our main model need to perform
# in our hyperparameter tuning 25 is best inference step number but you can play around with it loww number give poor result and high number give perfect result but it may take more time
# for 1 inference step it too around 6-7 mins on CPU and on GPU(Google CoLab) it takes around 1 min for 20 inference steps
# time may depend apone CPU and GPU (model)
