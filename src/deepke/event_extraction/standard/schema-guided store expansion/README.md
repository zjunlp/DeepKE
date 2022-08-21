
## Requirements
- Python 3.6
- Java 8

For other requirements on python packages, please look at ```requirements.txt```.

## How to Run the Code
The commands below run the rule-based event detection system on SW100.
```
$ sh download.sh
$ sh preprocess.sh
$ python wsd.py
$ python phrase.py
$ python rule.py
$ python ann2brat.py
```

Running `wsd.py` (word sense disambiguation) may take several hours.  If you
can successfully run the code, you should be able to get system output in the
[Brat standoff format](http://brat.nlplab.org/standoff.html) under `out/SW100`.
