import matplotlib.pyplot as plt
import sys

import dpfdata

dpfdata.dataSource = "vulkan"
d = dpfdata.load("20220602", 3)

if d is None:
  sys.exit(0) # Didn't find anything here

# print(f"Available data: {list(d.keys())}")
# print(f"Available scope plots: {list(d['Shot'].keys())}")

plt.figure()
plt.plot(d["Ag_Far"]["bin"], d["Ag_Far"]["counts"])
plt.ylabel("Counts")
plt.xlabel("Bin")
plt.title("Ag Far")

plt.figure()
mcd4 = d["Shot"]["MCD 4"]
plt.plot(mcd4["Time (s)"], mcd4["Amplitude (Volts)"])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (Volts)")
plt.title("MCD 4")

plt.figure()
idot1 = d["Shot"]["DPF Idot 1"]
plt.plot(idot1["Time (s)"], idot1["Amplitude (Volts)"])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (Volts)")
plt.title("Idot 1")

# ------------------------------------------------------------

dRogowski = dpfdata.load("20220110", 3)
rogT = dRogowski["rogowski"]["time"]
rogA = dRogowski["rogowski"]["A"]
rogB = dRogowski["rogowski"]["B"]
rogC = dRogowski["rogowski"]["C"]
plt.figure()
plt.plot(rogT, rogA)
plt.plot(rogT, rogB)
plt.plot(rogT, rogC)
plt.xlabel("Time")
plt.title("Rogowski Picoscope data")

plt.show()