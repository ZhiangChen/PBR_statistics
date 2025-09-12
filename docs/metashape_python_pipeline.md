# Metashape Python Pipeline
This tutorial provides step-by-step instructions for using Metashape v2.2 Python APIs to implement a structure-from-motion pipeline. Metashape is installed on a Red Hat server. The Metashape Python API requires Python 3.9. If Python 3.9 is not already installed on your system, you should install it before proceeding with Metashape setup.

## Activating Metashape on Server
The "License is in use" error is a known issue in Metashape Linux. This occurs because when the license is registered through the software interface, it gets stored in a temporary folder: `/var/tmp/agisoft/licensing`. This folder is deleted when the server reboots. To resolve this issue, the license must be installed in a permanent directory.

**Before registering the license**, you need to set up an environment variable to specify the license directory by adding the following line to your `.bashrc` file:
```bash
export AGISOFT_LICENSING_DIR=/home/zchen256/Documents/Agisoft_license
```

Verify that the environment variable is properly set:
```bash
printenv AGISOFT_LICENSING_DIR
```

Register the license using the command line:
```bash
./metashape.sh --activate XXXXX-XXXXX-XXXXX-XXXXX-XXXXX
```

After successful activation, you should see the license files created in the designated folder. 


## Setup Metashape Python
If Python 3.9 was not installed before installing Metashape, you may encounter errors when importing Metashape modules. In such cases, after installing Python 3.9 (after Metashape installation), you need to configure the Python environment paths properly. 

Create a bash script file named `python_metashape.sh` with the following content:
```bash
#!/bin/bash
export PYTHONHOME=$HOME/python39
export PYTHONPATH=$HOME/python39/lib/python3.9:$HOME/python39/lib/python3.9/site-packages:$HOME/metashape-pro/modules
export QT_QPA_PLATFORM=offscreen
$HOME/metashape-pro/metashape.sh "$@"
```

**Environment Variable Explanation:**
- The first two lines specify the Python paths. If you installed Python 3.9 before installing Metashape, these may not be necessary.
- `QT_QPA_PLATFORM=offscreen` is required when using remote SSH connections. Remove this line if you want to use Python in the Metashape GUI console.

**Usage for Remote SSH:**
When connecting via remote SSH, use the `-r` argument to run Python scripts:
```bash
./python_metashape.sh -r python_scripts/workflow_sampledataset.py
```

**Create a command:**
- Make the bash file executable: `chmod +x $HOME/metashape-pro/python_metashape.sh`
- Add its folder to PATH in `.bashrc`: `export PATH=$HOME/metashape-pro:$PATH`
- Source `.bashrc`
- Now you can run it anywhere, e.g., 
```
[zchen256@arius ~]$ python_metashape.sh -r metashape-pro/python_scripts/workflow_sampledataset.py
```

