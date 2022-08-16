# Running MONAI Apps in Nuance PIN

MONAI Deploy Apps can be deployed as Nuance PIN applications with minimal effort and near-zero coding.

This folder includes an example MONAI app, AI-based Spleen Segmentation, which is wrapped in the Nuance PIN API.
The Nuance PIN wrapper code allows MONAI app developer in most cases to deploy their existing MONAI apps in Nuance
without code changes  

## Prerequisites

Before setting up the 

Minimum software requirements:
- [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Nuance PIN SDK](https://www.nuance.com/healthcare/diagnostics-solutions/precision-imaging-network.html)

<div class="note">

Nuance PIN SDK does not require host installation to make the example app work. We will explore options in the [Quickstart](#quickstart) section 

</div>

(quickstart)=
## Quickstart


