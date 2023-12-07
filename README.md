# TP6 - Deep learning - Bioinformatique 2023 - Notes du programme

## Note #1 : instructions et projets
[Lien vers l'enoncé complet du TP et documentation du projet.](TP6_bin3002.md)

## Note #2 : clonage du projet et instructions pour l'utilisation de git sur les serveurs SENS 
Pour cloner ce repo sur SENS, utilisez ssh sur les FRONTAUX! (Sur dorsaux-sens ça ne marchera pas). Utilisez la commande: 

<code>git clone git@github.com:Kontrabass2018/bioinfo_deep_learning_2023.git</code>

Assurez-vous d'etre contributeur sur le projet et d'avoir configure vos cles ssh publique et prive.

## Note #3 : travailler en local avec virtualenv. 

Initializing the program, setting up environment variables (taken from [Source](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) )

To install venv via pip
```{bash}
python3 -m pip install --user virtualenv
```

Then, create  activate the environment (Only first time)
```
python3 -m venv env
```

**Activate environment (everytime to run)**

**On windows**

do this before activating. (in a powershell)*
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```
Then, to activate the environment. One of the options.
```
./env/Scripts/Activate.ps1
./env/Scripts/activate
```

**On Unix**
```
source env/bin/activate
```

**Install required packages (Only first time)**
```
python3 -m pip install -r requirements.txt
```

Then finally, to run the program, run :
```{python3}
python3 main.py 
```
The other commands will be explained.


