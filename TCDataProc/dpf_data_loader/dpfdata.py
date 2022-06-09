import pandas as pd

import sys
import re
import os
import io

PREFIX_VULKAN = r"\\10.100.160.3\DPF_Archive (D)\Data"
PREFIX_VDRIVE = r"V:\Programs (Team Folder)\DPF\Experimental Data\DPF Data"

paths = {
  "vulkan": {
    "data": f"{PREFIX_VULKAN}\\DPF Data",
    "rogowski": f"{PREFIX_VULKAN}\\Module Data",
    "mcd_tof": f"{PREFIX_VULKAN}\\Caen MCD Data",
    "thermocouple": f"{PREFIX_VULKAN}\\Thermocouple Data",
  },
  "vdrive": {
    "data": f"{PREFIX_VDRIVE}\\DPF Data",
    "rogowski": f"{PREFIX_VDRIVE}\\11 Module Data",
    "mcd_tof": f"{PREFIX_VDRIVE}\\06 Neutron Detectors\\Caen MCD Data",
    # XXX For some reason all files in the Thermocouple Data directory (below) are loaded as 0B...?
    "thermocouple": f"{PREFIX_VDRIVE}\\10 Temperature\\Thermocouple Data",
  }
}

dataSource = "vdrive"

def setDataSource(source):
  """
  @param: source, str, either "vulkan" or "vdrive"
  """
  global dataSource
  if source != "vdrive" and source != "vulkan":
    print('Error: data source needs to be either "vulkan" or "vdrive"')
  dataSource = source

def _parseFilename(filename):
  shotNum, fileType = -1, "scalar"

  mScalar = re.match(r'(\w+)\s*-\s*(\w+)_(\d+)_(\d+)\..*', filename)
  mScope = re.match(r'(\w+?)(\d+)_(\d+).*', filename)

  if mScalar is not None: # ...then this is a scalar
    s1, s2, shotNum, globalShotNum = mScalar.groups()
    name = f"{s1}_{s2}"
  elif mScope is not None: # ...then, maybe scope data?
    fileType = "scope"
    name, shotNum, globalShotNum = mScope.groups()
  else:
    return None, None, None, None

  return fileType, name, int(shotNum), int(globalShotNum)

def _parseScalar(filename):
  return pd.read_csv(filename, delim_whitespace=True, names=["bin", "counts"])

def _parseScope(filename, verbose=True):
  with open(filename) as fp:
    lines = fp.readlines()

  def splitPlotFile():
    allPlots, currPlot = [], []
    NUM_HEADER = 4
    if verbose:
      sys.stdout.write(f"Loading scope file ({len(lines)} lines)")
    for iLine, line in enumerate(lines[NUM_HEADER:]):
      if line.startswith("-----"):
        allPlots.append(currPlot)
        currPlot = []
        if verbose:
          sys.stdout.write(".")
          sys.stdout.flush()
      elif line.strip() != "":
        currPlot.append(line)
    if verbose:
      print() # output a newline

    return allPlots

  def readSeparatedScopeLines(allPlots):
    scope = {}

    I_TITLE, I_XLABEL, I_YLABEL, I_DATA = 1, 2, 3, 11
    iLabels = [I_TITLE, I_XLABEL, I_YLABEL]

    def extractTitle(line):
      lastItem = line.split(":")[-1].strip()
      prefix, title = re.match(r'(\d*_?)(.*)', lastItem).groups()
      return title # Discard the global shot number prefix

    for plotLines in allPlots:
      title, xLabel, yLabel = [extractTitle(plotLines[i]) for i in iLabels]
      dataStr = io.StringIO("".join(plotLines[I_DATA:]))
      data = pd.read_csv(dataStr, names=[xLabel, yLabel])

      scope[title] = data
      if verbose:
        print(f"Scope plot '{title}' parsed ({len(data)} points)")

    return scope

  allPlots = splitPlotFile()
  scope = readSeparatedScopeLines(allPlots)
  return scope

def _loadData(date, desiredShotNum, verbose):
  data = {}

  dataDir = f'{paths[dataSource]["data"]}\\{date}'
  if os.path.exists(dataDir):
    for filename in os.listdir(dataDir):
      fileType, name, shotNum, globalShotNum = _parseFilename(filename)

      path = os.path.join(dataDir, filename)
      if fileType is not None and shotNum == desiredShotNum:
        if verbose:
          print(f"Loading '{path}'...")
        data[name] = _parseScalar(path) if fileType == "scalar" else _parseScope(path, verbose)

  return data

def _loadThermocouple(date, desiredShotNum, verbose):
  data = {}

  dataDir = f"{paths[dataSource]['thermocouple']}"
  if os.path.exists(dataDir):
    for filename in os.listdir(dataDir):
      mTherm = re.match(r'(\w+)_(\d{8}).*', filename)
      if mTherm is not None:
        s1, dateStrBack = mTherm.groups()
        dateStr = dateStrBack[4:] + dateStrBack[:4]

        if date == dateStr:
          path = os.path.join(dataDir, filename)
          if verbose:
            print(f"Loading thermocouple data '{path}'...")
          dTherm = pd.read_csv(path)
          if len(dTherm) > 0:
            if verbose:
              print("Parsing thermocouple timestamps...")
            dTherm["t"] = pd.to_datetime(dTherm["Time"])
          data["thermocouple"] = dTherm

  return data

def _loadMCDTOFData(date, desiredShotNum, verbose):
  data = {}

  dataDir = f"{paths[dataSource]['mcd_tof']}\\{date}"
  if os.path.exists(dataDir):
    for filename in os.listdir(dataDir):
      mMCD = re.match(r'(\d+)_shot(\d+)_(\d+).*', filename)
      if mMCD is not None:
        dateStr, shotNum, globalShotNum = mMCD.groups()

        if int(shotNum) == desiredShotNum and date == dateStr:
          path = os.path.join(dataDir, filename)
          if verbose:
            print(f"Loading MCD/TOF data '{path}'...")
          data["mcd_tof"] = pd.read_csv(path)

  return data

def _loadRogowskiData(date, desiredShotNum, verbose):
  data = {}

  dataDir = f"{paths[dataSource]['rogowski']}\\{date}"
  if os.path.exists(dataDir):
    for filename in os.listdir(dataDir):
      mPico = re.match(r'(\w+?)(\d+)_(\d+)_(\d+)\..*', filename)
      if mPico is not None:
        s1, shotNum, globalShotNum, dateStr = mPico.groups()

        if int(shotNum) == desiredShotNum and date == dateStr:
          path = os.path.join(dataDir, filename)
          if verbose:
            print(f"Loading Rogowski data '{path}'...")
          data["rogowski"] = pd.read_csv(path)

  return data

def load(date: str, desiredShotNum: int, verbose=True):
  if dataSource == "vdrive" and not os.path.exists(PREFIX_VDRIVE):
    if verbose:
      print("Error: V-Drive is not mounted")
    return None
  elif dataSource == "vulkan" and not os.path.exists(PREFIX_VULKAN):
    if verbose:
      print("Error: Vulkan not accessible")
    return None

  data = { "date": date, "shotNumber": desiredShotNum }
  nBaseKeys = len(data.keys())

  dData = _loadData(date, desiredShotNum, verbose)
  dRogowski = _loadRogowskiData(date, desiredShotNum, verbose)
  dMCDTOF = _loadMCDTOFData(date, desiredShotNum, verbose)
  dTherm = _loadThermocouple(date, desiredShotNum, verbose)

  data = { **data, **dData, **dRogowski, **dMCDTOF, **dTherm }

  if len(data.keys()) > nBaseKeys:
    return data
  else:
    if verbose:
      print(f"No data found in for {date}, shot #{desiredShotNum}")
    return None