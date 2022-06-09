import dpf_data_loader.dpfdata

def test_filenames():
  params = [
    ["asdf", None],
    ["Ag - Near_2_7640.txt", "scalar"],
    ["Shot9_7647.txt", "scope"],
    ["Shot4_7494_Prefire.txt", "scope"],
  ]

  for filename, fileTypeRef in params:
    fileType, name, shotNum, globalShotNum = dpf_data_loader.dpfdata._parseFilename(filename)
    assert(fileType == fileTypeRef)