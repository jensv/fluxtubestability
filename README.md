# README #
This code determines the ideal MHD stability limit of current-carrying magnetic flux tubes / cylindrical screw pinches.
The code integrates Newcomb's Euler-Lagrange equation to determine the external stability of current profiles with a discrete core and skin current.

Flux tube experiments can be classified by the flux tube’s evolution in a configuration space described by a normalized inverse aspect-ratio $\bar{k}$ and current-to-magnetic flux ratio $\bar{\lambda}$. A lengthening current-carrying magnetic flux tube traverses this $\bar{k}$ - $\bar{\lambda}$ space and crosses stability boundaries. We derive a single general criterion for the onset of the sausage and kink instabilities in idealized magnetic flux tubes with core and skin currents. The criterion indicates a dependence of the stability boundaries on current
profiles and shows overlapping kink and sausage unstable regions in the $\bar{k}$ - $\bar{\lambda}$ space with two free parameters. Numerical investigation of the stability criterion reduces the number of free parameters to a single one that describes the current profile, and
confirms the overlapping sausage and kink unstable regions in $\bar{k}$ - $\bar{\lambda}$ space with an inverse dependence on current profile. A lengthening, ideal current-carrying magnetic flux tube can therefore become sausage unstable after it becomes kink unstable.

### Dependencies ###
python 2.7.12
numpy 1.11.2
scipy 0.18.1
matplotlib 1.5.3
seaborn 0.7.1
sqlite 3.13
gitpython 2.1.0

The dependencies can be installed with the anaconda python distribution.
gitpython can be installed with pip.

### Setup ###
Create three directories `figures`, `output`, `source`. Git clone or copy the repo into `source`.

### Important files ###
`recreate_paper_data.sh` recreates the data underlying figures 3 and 4 of the paper.
`skin_core_scanner.py` is a script file that runs a stability scan over $\bar{k}$-$\bar{\lambda}$ for an equilibrium defined by a core skin current profile of a given core current to total current ratio $\epsilon$. This file has a command line interface. Help can be accessed with `python skin_core_scanner.py --help`.
`newcomb.py` determines the external stability of a single equilibrium (single pair of \bar{k} \bar{\lambda}).
`paper_figures.py` recreates the figures in the paper. This file has a command line interface. Help can be accessed with `python paper_figures.py --help`. 

### Run ###
Run the bash_script `recreate_paper_data.sh`.
This will create a SQL database in `output` called `output.db` that keeps track of all runs and input parameters of `skin_core_scanner.py` and run `skin_core_scanner.py` three times to do a stability scan for three values of $\epsilon$.
The output data is stored in dated directories in `output`. 
`output.db` can be opened e.g. with the Firefox extension SQLite Manager and the run parameters can be inspected.
The paper figures can be generated with `python paper_fgiures ../output/[timestamp] ../output/[timestamp] ../output/[timestamp]`.
Where `[timestamp]` is the dated directory name. The order should be from oldest to newest date.

### newcomb.py description ###
Examines the equilibrium. If the equilibrium has a singularity, the Frobenius method is used to determine a small solution at an r > than instability. If the singularity is suydam unstable no attempt is made to calulate external stability. If there is no Frobenius instability power series solution close to r=0 is chosen or if the integration does not start at r=0 a given xi is used as boundary condition. Only the last interval is integrated. Xi and xi_der are plugged into the potential energy equation to determine stability.

### Contact ###
Jens von der Linden jensv@uw.edu


