# Welcome to Data-Drift-in-Machine-Learning

Code samples for [Machine Learning for Drifts and Shifts](todo) published by [Packt](https://www.packtpub.com/?utm_source=github).

Thank you for choosing our book, and for getting familiar with drifts in machine learning! 

This repository contains working versions of the snippets provided in the book to help you understand and address data drift challenges effectively.

## Chapters' examples

* [Chapter 4: Monitoring and Detecting Drifts](Chapter04)
* [Chapter 5: Concept Drift Detection and Handling](Chapter05)
* [Chapter 6: Continuous and Incremental Learning](Chapter06)
* [Chapter 7: Catastrophic Forgetting and Adaptive Learning](Chapter07)
* [Chapter 8: Model Re-training](Chapter08)
* [Chapter 9: Managing End-to-end Pipelines under the Presence of Drifts](Chapter09)
* [Chapter 10: Working with Datasets Partially Representing the Data Distribution](Chapter10)
* [Chapter 11: Learning Non-stationary Data](Chapter11)
* [Chapter 12: Dealing with Stationarities and Recurrences in Continual Learning](Chapter12)

## Setting Up Your Environment

Let's get started by setting up your environment. Follow Chapter #2 for specific details on installing all required packages. List of current requirements is present in [requirements.txt](requirements.txt) file.

1. **Create a Conda Environment:**
   ```bash
   conda create -n book python=3.10 -y
   conda activate book
   ```

1. **Clone the Repository and Install Requirements:**
   ```bash
   git clone https://github.com/PacktPublishing/Data-Drift-in-Machine-Learning.git
   cd tutorials
   pip install -r requirements.txt
   ```

1. **Installing [NannyML](https://www.nannyml.com/library):**

   To install NannyML on a Mac, follow these steps:

   1. **Install Homebrew** (if not already installed) from [Homebrew](https://brew.sh/).

   2. **Install libomp**:
      ```bash
      brew install libomp
      ```

   3. **Install LightGBM** via conda:
      ```bash
      conda install -c conda-forge lightgbm
      ```

   4. **Install NannyML** via pip:
      ```bash
      pip install nannyml==0.9.0
      ```

1. **Installing river-torch:**

Install as `pip install "river[torch]"`, we are currently using the 0.1.2 version.

## Contributing to Data-Drift-in-Machine-Learning

We welcome contributions from the community to enhance and expand Data-Drift-in-Machine-Learning. Here's how you can contribute:

- **Fork and Pull Requests:**
  Feel free to fork the Data-Drift-in-Machine-Learning repository and submit your contributions via pull requests. We acknowledge all contributors in our releases, and outstanding contributions may even earn you a spot on the Data-Drift-in-Machine-Learning conference website!

- **Bug Reports and Feature Suggestions:**
  We value bug reports, feature suggestions, and enhancements. Your feedback helps us improve Data-Drift-in-Machine-Learning and make it more robust and user-friendly.

- **Sharing Your Projects:**
  If you've built exciting projects or applications using Data-Drift-in-Machine-Learning, we'd love to hear about them! Share your links and experiences with the community to inspire others.

Your engagement enriches the Data-Drift-in-Machine-Learning community and contributes to its growth and development!

## License

This project is licensed under the GPL-3.0 licence 
