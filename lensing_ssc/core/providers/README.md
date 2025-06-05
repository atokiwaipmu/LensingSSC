# Providers

This directory contains provider classes that abstract access to external libraries or services. This allows for a consistent interface within `lensing_ssc` while enabling flexibility in choosing underlying implementations.

Examples of providers include:

-   **Base Provider**: An abstract base class for all providers.
-   **Factory**: A factory class for creating instances of different providers.
-   **Healpix Provider**: For operations related to HEALPix maps.
-   **Lenstools Provider**: For interfacing with the Lenstools library.
-   **Matplotlib Provider**: For generating plots using Matplotlib.
-   **Nbodykit Provider**: For interfacing with the Nbodykit library.
