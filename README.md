
Report
==========


**Abstract**

The aim of this report is to present an AI solution to predicting solar
flare occurrences, and forecasting 'all clear' situations for X-class
solar flares, and optionally for any class, within a 24-hour period.
This solution will be based on various sunspot properties collected in
the public UCI solar flare dataset (Bradshaw, 1989), we will be using a
Keras sequential model.

**Introduction**

Solar flares are explosions of electromagnetic radiation that happen on
the surface of the sun. They occur when magnetic energy that has been
collected in the solar atmosphere is abruptly emitted. The amount of
energy released could fuel the whole planet for 10 million years! Such
surge of energy can have calamitous effects since they can harm
satellites, GPS systems, power grids. They even have the potential to
create worldwide radio and electric blackouts. This radiation can also
negatively affect astronauts working outside the ISS.

Every 11 years our sun goes through a solar cycle which impacts its
activity, during this cycle the sun has its Solar Maximum in which a
large number of sunspots appear, and solar flares become more violent.
(Erickson, 2020). There are typically three stages to a solar flare. the
precursor, impulsive, and decay stages all together a solar flare can
last anywhere from a few minutes to a few hours. The primary danger of
solar flares; Nevertheless, comes in with their speed. Since solar
flares travel at the speed of light, they can be knocking on our doors
in about 8 minutes from detection using telescopes. [here is where
having an AI system that can predict whether a solar flare will happen
24 hours in advance comes in!]{.ul}

*"A big flare is a potential risk to our society; therefore, the
prediction of solar flares is crucial."* Kanya Kusano - Institute for
Space-Earth Environmental Research in Japan.

**Background**

Solar flares are classified based on their strength, going from A class
to B, C, M and finally X. between every class there is a 10 fold
increase in energy output and within each class, we can further define
the 9 folds in between using numbers. However since X class is the final
class we can have more numbers than 9. The most powerful flare ever to
be recorded happened during the previous solar maximum in 2003, the
sensors were overloaded and cut out at X28, but it was estimated to be
around X45.

The 2003 storm disrupted satellite TV and radio services, GPS systems,
damaged a Japanese scientific satellite beyond repair, sent several
deep-space missions into safe mode or complete shutdown, and destroyed
the Martian Radiation Environment Experiment aboard NASA\'s Mars Odyssey
mission. At the height of the storm, astronauts aboard the International
Space Station had to take cover from the high radiation levels. It also
caused airline communication problems which resulted in millions of
dollars in damages and send Antarctic science groups into full
communication blackout. (Nasa, 2007)

Having these tragic potential outcomes in mind, we decided to ignore A
and B classes in our AI solution since their effects are too small to
even be noticeable on Earth. Thus, we will only be considering C, M, and
X classes. Since X classes are the only class that could cause severe
and long-lasting effects, our solution will be based on that, but as an
additional option, we will also create a mode in our program to detect
solar flares of any class. The dataset we are using, the UCI- Solar
Flare Data Set, provides 13 attributes, 3 of them being class attributes
correlating to the number of recorded C, M, and X class flares in the
next 24 hours. The attributes go as follows:


<br/><br/>
![alt text](https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/0.PNG)

**Methodology and Data:**

To begin developing our AI solution we first need to import, transform,
visualize, and edit our data. Our data happens to be separated into
files \"flare.data1\" and \"flare.data2\" as they are presented in the
public dataset: nonetheless, a quick look at the description tells us
that "flare.data2" has had much more error correction applied to it, and
thus is more reliable. With such information at hand, we decided to try
to avoid flare.data1 as much as possible. With that said, we can now
move on to the first stop in our data-processing chain, importing.

1)  ***Importing:***<br/>
![alt text](https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/1.PNG)

Since our dataset didn't have any headers, we decided to add those
myself for good practice and future ease of use. we then created our
data frame by using the pandas read_csv function and reading in our
dataset. Notice as well, that we used the names attribute to name the
columns of our data frame with the headers that we created.

2)  ***Transforming:***<br/>
![alt text](https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/2.PNG)
After taking a look at our dataset we can see that our first three
columns contain string values rather than numeric ones. Numbers however
is the language of our machine learning algorithm, thus for the magic to
happen, we will need to feed it numbers and not letters. So, using an
ordinal encoder we did that transformation.

-   Class: (A,B,C,D,E,F,H) (0,1,2,3,4,5,6)

-   Largest Spot: (X,R,S,A,H,K) (0,1,2,3,4,5)

-   Spot Distribution: (X,O,I,C) (0,1,2,3)

Furthermore, we wanted to add an option for the user to predict whether
or not a solar flare is inclined to strike regardless of its class, we
decided to add one more column a "sum-class" which will be the sum of
the C, M, and X class results. we then turned the result into a Boolean
value. If the sum is greater than 0, the value of the sum column will be
1, otherwise, it is a 0.<br/>
![alt text](https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/3.PNG)

3)  ***Visualizing:***

Now that importing and transforming are done, we can move on to data
visualization to gain a more intuitive understanding of our data. To do
that we can use some python libraries like Matplotlib or Seaborn;
nonetheless, we decided that a better approach, in this case, was to use
a different data visualization tool known as PowerBI, which makes the
process extremely easy and straightforward. To do that; however, we
first needed to export our newly modified dataset, so we created a
function that does that and called it:<br/>
![alt text](https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/4.PNG)

Moving on, we imported the data into PowerBI and started creating some
art!
<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/5.PNG" width="60%">

The first report we created, and what we believe to be the most
important is the key influencers report. It analyzed how all the columns
of our dataset (other than the three attribute classes) impact the
\"sum-class\", and which of these columns tend to influence the
sum-class into increasing or decreasing.

This provided much valuable information, for instance, it showed that
when the Largest Spot is 1 or 2, the sum class is likely to increase.
What this means is that if the code of the largest sunspot detected 24
hours ahead is either R or S, we are much more likely to experience a
solar flare!
<br/>

<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/6.PNG" width="60%">

From the "Decrease" part of the key influencers report, we can also
observe that when activity is 1, we are much more likely to notice a
decrease in our sum class. Meaning that if that activity of our sunspot
was reduced (= 1) we are less likely to see a solar flare in the
upcoming 24 hours.

Many more useful reports were created, which gave me a lot of insight
and an intuitive understanding of the data. Here are two more to list a
few:
<br/>

<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/7.PNG" width="40%">
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/8.PNG" width="40%">

Finally, we also checked for multicollinearity and printed out a
detailed summary of our data in python:<br/>
![alt text](https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/9.PNG)
<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/10.PNG" width="60%">

<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/11.PNG" width="60%">

Now that we were done with data analysis, we could finally move on to
creating our sequential model and training it. There are numerous
variables to consider when it comes to creating our model, those include
but are not limited to the number of layers, the number of neurons on
each layer, the test/train split percentage, the number of epochs that
will be done in the fitting process, the batch size and many more. For
the sake of finding the most suitable values for each variable and
creating an optimal model, we decided to write a function that creates a
random model based on random variables. The idea behind that is that we
would create a large number of random models, test them, and then save
the model with the highest accuracy for future use:<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/12.PNG" width="60%">

As you can see, we first create a list of 4 random integers and 1 random
decimal with varying ranges,

Then we divide the data into testing and training subsets, with the
testing subset being anywhere from 10 to 30 percent. Furthermore, we
move on to the first step in creating a Keras model, **defining**.

In the defining stage, we define the model as a sequential model, we
also define the number of layers and the number of neurons on each
layer. When it came to the activation functions we chose Relu for the
inner hidden layers and sigmoid for the final layer since we wanted a
number from 0 to 1 as the outcome.

Next up comes **Compiling.** While compiling the model, 3 essential
features are defined

1)  Loss Function, which is used to find the error or loss in our
    learning process

2)  Optimization is used to find the error or loss in our learning
    process

3)  [Metrics]{.ul} define the criteria for determining how good of a
    model we created

we ended up using the binary_crossentropy loss function, the efficient
Adam optimizer, and the typical accuracy metric.

**Fitting:**

Fitting or training our model to our data was achieved by using the
model. fit() function; this function requires a few attributes.

1)  Our input data

2)  Number of epochs

3)  Optional: batch size and validation data

These values will be fed with the random numbers that we have created

Finally now having our function that creates random models ready, we
could create the training function which will be responsible for
creating a specified number of these random models and choosing the one
with the highest accuracy:<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/13.PNG" >

**Evaluation:**

In order to choose the best model, we needed to evaluate the model\'s
accuracy, this was done using the model.evaluate() function. After every
iteration we check if the new model's accuracy is higher than the one we
saved, if it is, we replace the old one with the new one. For saving, we
used the model.save() function.

**Prediction:**

we decided to split the prediction task into 3 primary functions:

Function 1: -<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/14.PNG">
This first function takes in an array of input values and returns a
rounded value of the model.predict() function. The model.predict()
function returns a number from 0 to 1 based on confidence levels that a
solar flare is bound to hit, After rounding we only return either 0 or
1, 0 meaning no solar flare, and 1 meaning a solar flare is predicted to
hit within 24 hours.

Functions 2 and 3: -<br/>
<img src="https://github.com/Ahmad-Alaziz/Solar-Flare-Detection-AI/blob/main/Report-Media/15.PNG" >

The functions above are simply for printing a more readable and easily
understandable response to the user. we also decided to create a 'mode'
for the program to make switching between training/testing and
predicting only X-classes and training/testing and predicting for any
class easier.

**Analysis and Discussions:**

After training our model to predict X class solar flares, we got 99+
accuracy. Now while that may seem extremely good at first glance, when
we analyze our data we can see an obvious issue. Our data simply
doesn\'t contain enough instances of when an X-class solar flare hit,
since they are quite rare, thus our model could keep on predicting that
no X-class solar flare is bound to hit and it would be right for the
vast majority of times. That is a primary reason why we decided to also
include the optional class prediction, which provided much more
realistic results with high 80\'s accuracy or low 90\'s.

**Conclusions and suggestions for future work:**

A major issue that we faced in this task goes back to the data set. The
data collected is quite old, it was donated on 1989-03-01, which means
it is about 30 years old. Taking that into consideration we can imagine
that the technology used back in the day might not have been as accurate
as of the technology we could use today. For future works, we would love
to work on a more modern dataset with more accurate data, for us to
receive more accurate results.

References
==========

Bradshaw, G. (1989, 03 01). *archive.ics.uci.edu*. Retrieved from
http://archive.ics.uci.edu/ml/datasets/solar+flareBrownlee, J. (2019,
July 24). Retrieved from machinelearningmastery.com:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/Brownlee,
J. (2020, July 03). *machinelearningmastery.com*. Retrieved from
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/Erickson,
N. O. (2020, Dec 17). Retrieved from spaceplace.nasa.gov:
https://spaceplace.nasa.gov/solar-cycles/en/\#:\~:text=The%20Sun\'s%20magnetic%20field%20goes%20through%20a%20cycle%2C%20called%20the,and%20south%20poles%20switch%20places.&text=Giant%20eruptions%20on%20the%20Sun,increase%20during%20the%20solar%20cycle.M
M Faniqul, I., Rahatara, F., Rahman, S., & Yasmin, B. (2020, 07 12).
*UCI Early Stage Diabetes Risk Prediction Dataset*. Retrieved from
https://archive.ics.uci.edu/:
https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.Nasa.
(2007, August 9). Retrieved from Nasa, gov:
https://www.nasa.gov/mission_pages/sunearth/news/X-class-flares.html\#:\~:text=Solar%20flares%20are%20classified%20according,M%20and%20X%2C%20the%20largest.&text=Solar%20flares%20are%20giant%20explosions,high%20speed%20particles%20into%20space.Stephanie,
W. (2020, Feb 26). Retrieved from www.healthline.com:
https://www.healthline.com/health/diabetes\#:\~:text=Diabetes%20mellitus%2C%20commonly%20known%20as,the%20insulin%20it%20does%20make.
