# RubikScrapTool

<h3> Web Scrapping Tool / Data Analytics powered by GPT-3 / 3D Visualization of High-Dimensional Data </h3>
</br></br>

### Prerequisites

You should have Python 3.8 or above already installed.


### Installation

1. In your cmd terminal, go into folder where your want to install your venv:
  ```sh 
  cd PATH_TO_FOLDER
   ```
   
2. Create a Virtual Environment in the root folder
  ```sh
  python -m venv .venv
  ```

3. Clone the repo from GitHub:
   ```sh
   git clone https://github.com/Ashoka74/RubikScrapTool.git
   ```
4. Install required packages
   ```sh
   pip install -r requirements.txt
   ```
   
5. Add your own OpenAIAPI in 'scrappingtool\Tweeter\views.py' (line 37)
   
6. From the scrappingtool subfolder, run the following command :
   ```sh
   manage.py runserver
   ```
7. Open the url and launch your queries


### Things to know

1. Your first request will install some models on your machine so it will take a considerable amount of time

2. Each request can take a few minutes to complete depending on your computer's capacities
