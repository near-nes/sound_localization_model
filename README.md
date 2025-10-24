# Sound Localization

## External Data
**IRCAM Dataset [HRTFs]**  
Pre-installed HRTFs from 3 different human subjects are included.  
[Download here](http://recherche.ircam.fr/equipes/salles/listen/download.html)  

---

## Simulation
To run a simulation:  
1. Modify the input parameters in `main.py`:
   - Sound stimulus  
   - Binaural cue scenario  
   - Cochlea model  
   - Neuronal parameters  
2. Execute the script.  

Logs will be generated to monitor the simulation and help identify any errors.

---

## Results
- After generating ANF spike trains, a directory `ANF_SPIKETRAINS` will be created inside `Data` to store them. These can be reused in future simulations since spike train generation is computationally intensive.  
- A `.pic` file will be saved in the `RESULTS` directory, which is automatically created to store all output data.  

---

## Plots
Jupyter Notebooks to replicate the figures from the paper are included in `src/plot` repository.
