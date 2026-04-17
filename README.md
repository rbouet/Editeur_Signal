# EEG Spike Editor (PySide6 + PyQtGraph)

Editeur interactif de signaux EEG multi-canaux permettant :        
- la visualisation     
- le filtrage    
- la navigation temporelle     
- l’édition de marqueurs (spikes) définis par channel     

Ce projet est pensé comme un outil proche des logiciels EEG professionnels (BrainVision, EEGLAB, etc.), tout en restant léger, scriptable et modifiable pour la recherche. Les inputs sont par conséquent basiques et indépendnt de tout format. Il convient donc à l'utilisateur d'extraire :    
- signals : matrice, channel X time, amplitudes des signaux      
- times : vecteur, temps pour chaque sample de signal       
- channel_names : liste, channel's names     
- markers_df : pandas.DataFrame, 2 colonnes, "channel" et "sample" respectivement la localisation spatiale (channel name) et temporelle (en sample)        

# Lancement

Un exemple de lancement (lancement_editeur_eeg_spike.ipynb) indique comment visualiser un tracé au format Micromed (TRC) et des marqueurs provenant du soft Delphos.    

Voici un exmple de lancement :      

import sys
sys.path.insert(0, '/Users/romain/Study/Rheins/Thalamus_Git/Thalamus_Rheins_2025/Utils')
sys.path.insert(0, '/Users/romain/Study/plateformeintra/Editeur_Signal')

from eeg_spike_editor_qt import launch_editor

editor = launch_editor(
    signals = signals,
    times = times,
    channel_names = channel_names,
    markers_df = markers_df,
    window_sec = 20,
    n_display = 40
    )
           


# Dépendances
pip install numpy pandas scipy PySide6 pyqtgraph mne mne-connectivity antropy neurokit2 scikit-learn pyinform
pip show numpy pandas scipy PySide6 pyqtgraph mne mne-connectivity antropy neurokit2 scikit-learn pyinform

Python ≥ 3.9     
PySide6      
pyqtgraph     
numpy     
pandas    
scipy    
IPython    
