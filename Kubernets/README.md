# Kubernetes

## Table of Content

- [Kubernetes](#kubernetes)
  - [Table of Content](#table-of-content)
  - [History and Motivation](#history-and-motivation)
    - [**2000s: Traditional Deployment Era**](#2000s-traditional-deployment-era)
    - [**2010s: Virtualized Deployment Era**](#2010s-virtualized-deployment-era)
    - [**2020s: Container Deployment Era**](#2020s-container-deployment-era)
  - [Technology Overview](#technology-overview)
    - [Planes and Nodes](#planes-and-nodes)
    - [Kubernets System Components](#kubernets-system-components)

## History and Motivation

### **2000s: Traditional Deployment Era**

During the 2000s, we experienced what is known as the "Traditional Deployment Era". This period was characterized by:

- **On-premises Deployments**: Companies managed their own data centers or used colocation services.
- **Teams of Sysadmins**: Dedicated teams of system administrators were responsible for provisioning and managing fleets of servers, which was often a labor-intensive and immature process.
- **Bare Metal Servers**: Applications ran directly on physical servers.
- **Monolithic Architecture**: The prevalent architectural style was monolithic, where applications were built as single, indivisible units.
- **Homegrown Monitoring Tools**: Monitoring and managing applications required custom-built tools due to the lack of standardized solutions.

### **2010s: Virtualized Deployment Era**

The 2010s marked the transition to the "Virtualized Deployment Era". Key developments during this time included:

- C**loud Computing**: The advent of cloud computing allowed Virtual Machines (VMs) to be created and destroyed in minutes, providing greater flexibility and scalability.
- **Configuration Management Tool**s: Tools like Puppet and Chef became popular for managing infrastructure as code, simplifying the configuration and management of large-scale deployments.
- **Manual Bin-Packing**: Applications were manually allocated to VMs, optimizing resource usage but still requiring significant manual effort.
- **Improved Tooling**: The emergence of better tooling made it practical to manage a larger number of applications and cloud resources.
- **Challenges with Scale**: Despite the improvements, managing large numbers of cloud resources remained a significant challenge.

### **2020s: Container Deployment Era**

In the 2020s, we entered the "Container Deployment Era", which brought about transformative changes in how workloads are managed:

- **Workload Orchestrators**: Tools like Kubernetes enabled treating clusters of machines as a single resource, simplifying management and scaling.
- **Standard Interfaces and Utilities**: These orchestrators provided a range of utilities and interfaces to handle:
- **Efficient Scheduling**: Optimally distributing workloads across instances.
- **Health Checks**: Monitoring the health and status of applications.
- **Service Discovery**: Automating the detection of service locations within the cluster.
- **Configuration Management**: Standardizing the way configurations are managed and applied.
- **Autoscaling**: Automatically adjusting the number of running instances based on demand.
- **Persistent Storage**: Managing storage that persists beyond the lifecycle of individual containers.
- **Networking**: Ensuring reliable and scalable networking between services.

## Technology Overview

### Planes and Nodes

<img src="https://courses.devopsdirective.com/_next/image?url=%2Fkubernetes-beginner-to-pro%2F02-01-control-and-data-planes.jpg&w=1920&q=75" align=center />


The first concepts to understand with regard to kubernetes are:
- **Node**: A node is a worker machine in Kubernetes, previously known as a minion. A node may be a VM or physical machine, depending on the cluster. Each node has the services necessary to run pods and is managed by the control plane.
- **Control Plane**: A subset of nodes in the cluster dedicated to performing system tasks. Nodes that are part of the control plane are referred to as "control plane nodes".
- **Data Plane**: A subset of nodes in the cluster dedicated to running user workloads. Nodes that are part of the data plane are referred to as "worker nodes".

### Kubernets System Components

<img src="https://courses.devopsdirective.com/_next/image?url=%2Fkubernetes-beginner-to-pro%2F02-02-k8s-architecture.jpg&w=1920&q=75" align=center />

Kubernets is comprised of many smaller components
- **etcd**: Key-value store used for storing all cluster data. It serves as the source of truth for the cluster state and configuration.
- **kube-apiserver**: The front end for the Kubernetes control plane.
- **kube-scheduler**: Schedules pods onto the appropriate nodes based on resource availability and other constraints.
- **kube-controller-manager**: Runs controller processes. Each controller is a separate process that manages routine tasks such as maintaining the desired state of resources, managing replication, handling node operations, etc...
- **cloud-controller-manager**: Integrates with the underlying cloud provider (if running in one) to manage cloud-specific resources. It handles tasks such as managing load balancers, storage, and networking.
- **kubelet**: An agent that runs on each worker node and ensures that containers are running in pods and manages the lifecycle of containers.
- **kube-proxy**: This network proxy runs on each node and maintains network rules to allow communication to and from pods.