# Penalty law recognizer
### Written in Python, JavaScript, HTML, CSS, Node.js using Express.
***This project was done as part of Digital Humanities election course***

As part of collaboration with the department of Justice in Israel, We assist in the process of switching legal documents and documenting method to digital.

#### <ins> The goal of this project:

determine if a given section in law is a penalty section.

#### <ins> How does it work:

You may insert the content of the section (in Hebrew) on the website we've built.

Our program will do the following calculation: 

1. Create data set from the given data [tag, section_content]
2. Spliting the data into train and test sets (80%-20%)
3. Create a bag of words
4. Create a classifier by training the model

Clicking the button will return the classification of the given section (penalty / not penalty)

#### <ins> Running the project: </ins>

1. Run this command in your terminal:

``` git clone https://github.com/danadvo/penal-law-recognizer.git ```

2. Install Python, Sqlearn, Pandas, numPy libraries

3. Build and run the following command in your terminal:

``` node server.js ```

4. Open your browser on: http://localhost:3000/ 

You should see this:

![screen](https://iili.io/qFjLrv.png)

5. Insert the content of the section and press thr button
