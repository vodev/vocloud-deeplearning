
# VO-Digger -- Virtual Observatory Digger

Tool for classification data using deep neural networks on GPU (CUDA).



## Installation



## Usage

The inputting data are expected to have <float> values and <int> label.


### config.json


#### Configuration for CSV files

If the data is CSV then you can set which column (by 0-based index) is the label and which columns
(set as 0-based all-inclusive interval) hold the data.

```
data : {
	type: "csv",
	label: (int) index of the label column (0-based),
	start: (int) index where data columns start (inclusive, 0-based),
	end:  (int) index where data columns stop (inclusive),

	# now the optional parameters follow
	delimiter : (char) [default ','] one-character delimiter for csv data
	header: (bool) [default true] indicating if a header is in the data file

}
```
