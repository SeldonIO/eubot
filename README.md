# eubot
Machine learning classifer for the EU Referendum

# usage
No compliation for the code is necessary, but a few libraries need to be installed before the python becomes runnable (Python >3 is also required):

```
flask
flask-cors
pandas
numpy
scipy
scikit-learn
nltk (nltk.download() must also be exectured from the python terminal)
```

After installing the required libraries, run backend/FlaskPredictor.py. This will run a flask server by default on *http://localhost:5000* - this is changeable in the last line of the python.

Next, you will need to put the frontend files on a localhost server - if you will not be using port 80, FlaskPredictor.py's line 39 must be changed to reflect this - ```CORS(app, origin="http://localhost")``` --> ```CORS(app, origin="http://localhost:XXXX")```. If you have no idea how to do this, you can use programs such as XAMPP or MAMP - just move the frontend files into the htdocs folder, and add the line ```ServerName 127.0.0.1:80``` (this can be changed if you are not using port 80 or localhost) to the file *httpd.conf* - if this is not done, the jQuery AJAX requests will be made unbearably slowly.

After doing this, you should be good to go! Just navigate to http://localhost, and the frontend should work and communicate with the backend.

# data sources
The majority of the data was found from Google searches and random news sources - if you want to add any more data, just add it to the relevant folder in /backend - however, it needs to be encoded in utf-8 - this is what the python will be expecting, and will throw an error if not encoded correctly.

I should also give a special mention to the github repo https://github.com/johnrees/eu-referendum - data can be found here which was used for initial testing.

# license
The code uses an apache-2 license, which can be found in the file LICENSE-2.0.txt, and at the top of all code files.
