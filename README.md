# ABZ 2025 Case Study

## Training and testing of agents

Setup:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

The agent for the Highway Environment can be trained or tested via command line:

```bash
# To train a new model (save under models/highway_env):
python highway_agent.py train


# Loads the trained model and runs and renders episodes over its behaviour:
python highway_agent.py test
```


## Scenarios


In the following, we provide some scenarios.
The controlled vehicle (ego vehicle) is marked in red.


## Scenario 1


| <img src="images/Scenario1_1.png" alt="Scenario 1.1" width="150%">                         | <img src="images/Scenario1_2.png" alt="Scenario 1.2" width="150%"> | <img src="images/Scenario1_3.png" alt="Scenario 1.3" width="150%"> |
|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| A vehicle approaches another vehicle in the same lane from behind, the distance decreases. | The rear vehicle brakes.                                           |The rear vehicle continues braking to maintain the safety distance.|


## Scenario 2


 ![Scenario 2](images/Scenario2_1.png)                                                      | ![Scenario 2](images/Scenario2_2.png)        | ![Scenario 2](images/Scenario2_3.png)                                    
|--------------------------------------------------------------------------------------------|----------------------------------------------|--------------------------------------------------------------------------|
| A vehicle approaches another vehicle in the same lane from behind, the distance decreases. | The rear vehicle switches to the right lane. | The rear vehicle completes switching to the right lane and accelerates.  |


## Scenario 3

 ![Scenario 3](images/Scenario3_1.png)                   | ![Scenario 3](images/Scenario3_2.png)                                              | ![Scenario 3](images/Scenario3_3.png)       
|---------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------|
| There is another vehicle on another lane further right. | The rear vehicle switches to the right lane, as the safety distance is maintained. | The rear vehicle continues driving forward. |


## Scenario 5

 ![Scenario 5](images/Scenario5_1.png)                   | ![Scenario 5](images/Scenario5_2.png)                                                         | ![Scenario 5](images/Scenario5_3.png)       
|---------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------|
| There is another vehicle on another lane further right. | The rear vehicle cannot switch to the center lane, as the safety distance is not maintained.  | The rear vehicle continues driving forward. |