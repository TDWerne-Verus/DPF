# DPF Data Loader

Loads DPF data by date/shot number directly from the V-Drive for analysis. See 
`plotData.py` for usage example (or see below).

## Install

To install, run `python setup.py install --user` (`--user` flag is optional, to 
install to a non-system root directory).

Note the `pandas` library is required. If it is not installed automatically by 
the above command, use `pip install pandas --user` to install.

## Usage

```python
import dpf_data_loader

# Data will be a dictionary containing scalars and scope data
data = dpf_data_loader.load("20220504", 3)

idot1Amplitude = data["Shot"]["DPF Idot 1"]["Amplitude (Volts)"]
agFarBin = data["Ag_Far"]["bin"]
agFarCounts = data["Ag_Far"]["counts"]
# ...etc
```

An optional third parameter to `load` is `verbose`, which when `False` silences 
console output (`True` by default).

Data can also be directly loaded from Vulkan, rather than the V-Drive, using 
the following:

```python
dpf_data_loader.setDataSource("vulkan")
```

Use `dpf_data_loader.setDataSource("vdrive")` to set the data source back to 
the V-Drive (it is set to this by default). Note files sync'd from the V-Drive 
are cached locally, so they only need to be downloaded once, on the first 
viewing --- unlike on Vulkan, where files must be re-downloaded every time.

Data available in the returned dictionary are dynamically populated depending 
on the data present in the directory and scope file, but seem to often consist 
of:

    data
        Ag_Far
            bin
            counts
        Ag_Near
            bin
            counts
        XL_Be1
            bin
            counts
        XL_Be2
            bin
            counts
        Shot
            DPF DT Near TOF 1
                Time (s)
                Amplitude (Volts)
            DPF I
                Time (s)
                Amplitude (Volts)
            DPF Idot 1
                Time (s)
                Amplitude (Volts)
            DPF Idot 2
                Time (s)
                Amplitude (Volts)
            DPF V
                Time (s)
                Amplitude (Volts)
            MCD 1
                Time (s)
                Amplitude (Volts)
            MCD 2
                Time (s)
                Amplitude (Volts)
            MCD 3
                Time (s)
                Amplitude (Volts)
            MCD 4
                Time (s)
                Amplitude (Volts)
            DPF DT Near TOF 2
                Time (s)
                Amplitude (Volts)
            DPF DT Far TOF 1
                Time (s)
                Amplitude (Volts)
            DPF DT Far TOF 2
                Time (s)
                Amplitude (Volts)
        rogowski
            time
            A
            B
            C
            ... (etc, all other Picoscope channels)
        mcd_tof
            SecA-Ch1
            SecA-Ch2
            SecA-Ch4
            SecA-Ch5
            SecA-Time(ns)
            SecB-Ch1
            SecB-Ch2
            SecB-Ch4
            SecB-Ch5
            SecB-Time(ns)
            SecC-Ch1
            SecC-Ch2
            SecC-Ch4
            SecC-Ch5
            SecC-Time(ns)
        thermocouple:
            Time
            #0: CH0 [°C]
            #1: CH1 [°C]
            #2: CH2 [°C]
            #3: CH3 [°C]
            #4: CH4 [°C]
            t

Use `data.keys()` and `data["Shot"].keys()` to view available data for your 
particular case.
