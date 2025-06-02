# BirdNET Geomodel
Spatiotemporal species range prediction for detection post-filtering

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/birdnet-team/geomodel.git
    cd geomodel
    ```

2.  **Set up a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Google Earth Engine Setup:**

    *   **Sign up for Google Earth Engine:** If you don't have one already, sign up for a Google Earth Engine account at [https://earthengine.google.com/](https://earthengine.google.com/).

    *   **Authenticate:** Authenticate the Earth Engine Python API using the following command:

        ```bash
        earthengine authenticate
        ```

        This will open a browser window where you can log in with your Google account and grant the necessary permissions.

    *   **Initialize Earth Engine in your script:**
        Ensure you have the following lines in your Python script to initialize Earth Engine:

        ```python
        import ee
        ee.Initialize()
        ```

5.  **Data:**

    We use iNaturalist and eBird observations as training data. You can download the datasets from the following links (GBIF Darwin Core format):
    *   [iNaturalist Observation Dataset](http://www.inaturalist.org/observations/gbif-observations-dwca.zip)
    *   [eBird Observation Dataset](https://hosted-datasets.gbif.org/eBird/2023-eBird-dwca-1.0.zip)

    **Make sure to appropriately citate the sources of these datasets in your work.**

    After copiyng the datasets to your working directory, create a `.env` file and specify the path:

    ```bash
    WORKING_DIRECTORY="/path/to/your/working/directory"
    ```

## Usage

### geoutils.py

This script generates a grid of environmental data and plots it. It uses Google Earth Engine to retrieve environmental information for each grid cell.

1.  **Set the `WORKING_DIRECTORY` environment variable:**

    Create a `.env` file in the root directory of the project and set the `WORKING_DIRECTORY` variable to the path where you want to store the generated data:

    ```bash
    WORKING_DIRECTORY="/path/to/your/working/directory"
    ```

2.  **Run the script:**

    ```bash
    python utils/geoutils.py --grid_step_km 50 --ocean_sample_chance 0.01 --plot_column elevation_m
    ```

    **Arguments:**

    *   `--grid_step_km`: Step size for the grid in kilometers (default: 100).
    *   `--ocean_sample_chance`: Probability of sampling an ocean point (default: 0.1).
    *   `--plot_column`: Column to plot (default: `elevation_m`). Set to `None` to skip plotting.

    Results will be saved in the working directory, and plots will be generated in the `plots` directory.

## Citation
If you use this code in your research, please cite as:

```bibtex
@article{birdnet-geomodel,
  title={The BirdNET Geomodel: Spatiotemporal species range prediction for detection post-filtering},
  author={Kahl, Stefan and Lasseck, Mario and Wood, Connor and Klinck, Holger},
  year={2025},
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Funding

This project is supported by Jake Holshuh (Cornell class of ´69) and The Arthur Vining Davis Foundations.
Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072).
The German Federal Ministry for the Environment, Nature Conservation and Nuclear Safety contributes through the “DeepBirdDetect” project (FKZ 67KI31040E).
In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
