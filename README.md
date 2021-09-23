# Hive-HQ-backend

Project for Hack the North 2021.  
Also available at: https://devpost.com/software/hive-hq


## 💡 Inspiration
As society and public services reopen, stores and other areas are becoming crowded again. However, with the threat of COVID-19 variants and increasing case counts looming over society, it is still vital to be mindful of crowded areas. This is where Hive-HQ comes in – it is an AI-powered, multi-target tracking system that identifies common hotspots and bottlenecks where people crowd together in unsafe situations. We set out to build a tool to help businesses monitor their premises and prevent the spread of COVID-19, creating a safer world for everyone.

## ❓ What it does
Hive-HQ is a full suite of enterprise management tools used to monitor and improve a location’s safety with reference to COVID-19 guidelines. Based on a user-provided floor plan and data processed from security cameras, it presents live statistics of the number of people in a given store or public area, creates a heat map of foot traffic within a certain area, and provides recommendations based on said heat map to improve customer flow. There is functionality to add, remove, and configure any number of cameras in the system.

The live heat map displays the current number of customers throughout a location using a gradient coloured scale to indicate how dangerous a given situation potentially may be (blue meaning safe to red meaning unsafe). Based on this heatmap, a business can generate and implement customized recommendations that change the layout of the area (tables, displays, etc.) in order to improve circulation and eliminate bottlenecks and regions that see many customers passing through. By providing tools to help enterprises evaluate and improve the health safety of their physical locations, Hive-HQ is building safer environments for businesses and their customers.

## 💼 Business viability
Hive-HQ builds on top of existing infrastructure. Security cameras around an environment are perfect for this application. This means there’s little to no upfront cost to deploy this solution. We also leverage cloud computing, allowing for deployment everywhere and for even low-power devices, such as laptops, Chromebooks, and phones, to utilize our app. Most importantly, Hive-HQ gathers essential data that many clients would desire. Concentrations on the heatmap are able to reveal where people tend to gather, the popularity of a product or display, and possible improvements to the layout of a store to be more “COVID safe”.

Our vision for a user-friendly and easily-implemented service that monitors and provides important information about a given environment and its COVID-19 risk. This vision is unique and has a clear direction, with few if any current competitors in the market. And our concept for such a service comes at the right time. As society and the economy reopen, and as more consumers are engaging with businesses' physical locations, it is more important than ever to ensure that case counts remain low and that everyone stays safe. Our model, currently focused on potential applications for COVID-19, can also be easily reconfigured to maximize success in a post-COVID world. For example, Hive-HQ may be used to determine the success of a product or marketing campaign based on the number of consumers that have gathered or passed through a certain area. Overall, Hive-HQ is highly viable as a business model, with a unique identity, clear relevance for our time, and the potential for tremendous impact.

## 🧰 How we built it
**YOLOv5:** A brand new advanced AI architecture used to identify and track people within the camera streams. The video stream is then annotated and sent through the internet to the web client where everything is managed.

**PyTorch & OpenCV:** PyTorch was the core that powered our AI architecture. After the video stream was processed by YOLOv5, it was annotated using OpenCV and made accessible as a web server.

**Matrix Perspective Transformations:** Used advanced mathematical operations to transform 3D coordinates from multiple cameras onto a 2D floor plan.

**Google Cloud:** Using a P100 GPU on Google Cloud, we were able to leverage immense computing power to process multiple streams at once. This allowed deployment of our solution on low spec machines and virtually anywhere with an internet connection (and access to a camera).

**React.js:** We created the client side of our web app using React.js and JSX based on a high-fidelity prototype we created using Figma, incorporating a dashboard, camera configuration tools, and interaction with our custom API.

## 🤔 Challenges we ran into
* We faced a myriad of challenges when trying to choose a cloud provider with the specifications we needed and with a good interface to use. We decided to go with Google Cloud Platform after a couple hours of trying to figure out Microsoft Active Directory.
* GCP required us to submit an application to increase our GPU quota, otherwise it would not have been possible to use it.
* GPU drivers created compatibility issues with PyTorch. It took hours to identify the issue and enable CUDA for YOLOv5.
* When starting to work on our AI engine, we had to decide on whether or not to use a custom model, and in the interest of time, we decided to use a pre-trained model. We then had to decide which model we should use. We opted to use the new YOLOv5, as we had used YOLOs for previous projects.
* We had difficulty looking for realistic and applicable sample footage for our testing.

## 🏆 Accomplishments that we're proud of
* Creating a fully-functioning app with bi-directional communication between the AI server and the client.
* Our implementation of various mathematical functions that are optimized to run in Python, allowing the server to maintain a sustained frame rate of around 30 fps.

## 🧠 What we learned
* Perspective transformation algorithms and maths.
* State management in React.
* How to use Google Cloud Platform provisioning features.
* Camera servers and creating a custom API for communication between server and client.
* Team collaboration and communication.

## 👉 What's next for Hive-HQ
There are multiple things we would like to further implement in our project:

* Custom-trained AI model optimized for detecting people from security cameras.
* Heat map analysis and custom recommendations overlaid on the floor plan.
* More analysis tools for direction of travel.
* Auto-identify where traffic tends to merge.
* More intuitive dashboard displaying more information.
