# Spatial-Temporal Graph Neural Networks for Groundwater Data


This repository contains the code and resources for the research paper ["Spatial-Temporal Graph Neural Networks for Groundwater Data"](https://rdcu.be/dXsMm) (ML Taccari, H Wang, J Nuttall, X Chen, PK Jimack), published in *Scientific Reports* - Nature (DOI: [10.1038/s41598-024-75385-2](https://doi.org/10.1038/s41598-024-75385-2)).

## Abstract
This paper introduces a novel application of spatial-temporal graph neural networks (ST-GNNs) to predict groundwater levels. Groundwater level prediction is inherently complex, influenced by various hydrological, meteorological, and anthropogenic factors. Traditional prediction models often struggle with the nonlinearity and non-stationary characteristics of groundwater data. Our study leverages the capabilities of ST-GNNs to address these challenges in the Overbetuwe area, Netherlands.

We utilize a comprehensive dataset encompassing 395 groundwater level time series and auxiliary data such as precipitation, evaporation, river stages, and pumping well data. The graph-based framework of our ST-GNN model facilitates the integration of spatial interconnectivity and temporal dynamics, capturing the complex interactions within the groundwater system. Our modified Multivariate Time Graph Neural Network model shows significant improvements over traditional methods, particularly in handling missing data and forecasting future groundwater levels with minimal bias. The model's performance is rigorously evaluated when trained and applied with both synthetic and measured data, demonstrating superior accuracy and robustness in long-term forecasting. The study's findings highlight the potential of ST-GNNs in environmental modeling, offering a significant step forward in predictive modeling of groundwater levels.

## Repository Structure
- `requirements.txt` - Lists all necessary packages to run the code.
- `/data` - Contains input, preprocessed, and simulated data.
- `/data_preprocessing` - Scripts for data preparation.
- `/models` - Contains the model architecture.
- `/utils` - Utility functions.

## Citation
If this work aids your research, please cite our paper:

```bibtex
@article{Taccari2024,
  author = {Maria Luisa Taccari and He Wang and Jonathan Nuttall and Xiaohui Chen and Peter K. Jimack},
  title = {Spatial-Temporal Graph Neural Networks for Groundwater Data},
  journal = {Scientific Reports},
  volume = {14},
  number = {24564},
  year = {2024},
  doi = {10.1038/s41598-024-75385-2},
  url = {https://rdcu.be/dXsMm}
}
