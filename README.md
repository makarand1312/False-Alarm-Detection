## False Alarm Detection

# Problem Statement-

***This project was made for a chemical industry which had sensors installed in various parts of the factory to detect H2S gas which is hazardous to health. Every time one or multiple sensors detected the H2S leak, an emergency alarm rings to alert the workers. For every alarm, the industry calls a team which sanitizes the place and checks for the leak and this was a big cost to the company.***

A few of the alarms that ring are not even hazardous. The company gave us the data for each alarm with a final column stating the alarm was dangerous or not.

**[ Ambient Temperature |	Calibration(days)	| Unwanted substance deposition (0/1)	| Humidity (%)	| H2S Content(ppm)	| Dangerous (0/1) ]**

The data was first pre-processed and analysis libraries like **Numpy** and **Pandas** were used to make it ready to be utilized by a machine learning algorithm.

Problems like standard scaling, categorical data and missing values were handled with appropriate techniques.

Then, we used **Logistic model** to make a classifier with first five column as independent columns and dangerous column as **dependent/target column.**

Now whenever, there is a leakage and the alarm rings, the data is sent to us and we predict if it is dangerous or not. If found dangerous then only the team is called to sanitize the place and fix the leak. This saved a lot of money for the company. 
