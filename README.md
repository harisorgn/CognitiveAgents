This repository contains a julia module that fits behavioural data from three tasks [link to CALM study and repository when published]. 

The data is binary choices and in the case of the faces matching task, it also contains response times. The fitting method is an interior-point algorithm implemented in MadNLP.jl run via the JuMP.jl optimization interface.

The `/analysis_scripts` folder contains three scripts (`run_CL.jl` , `run_CM.jl` and `run_faces.jl`), each one running the entire workflow for fitting one out of the three models for the respective behavioural task. These scripts contain everything that can be done using this interface; fit parameters, extract regressor timeseries from the model and some useful figure functions.
