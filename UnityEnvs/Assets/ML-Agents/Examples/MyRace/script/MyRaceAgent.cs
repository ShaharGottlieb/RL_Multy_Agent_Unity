using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class AgentRewardSystem
{
    // CONSTANTS - REWARD CONFIGURATIONS //
    private const float velocityRewardCostant = 0.05f;
    private const float collisionPen = -0.4f;
    private const float checkpointReward = 0.2f;
    private const float stayStillPen = -0.01f;
    private const float stayStillLimit = 0.3f;
    //      PRIVATE VARIABLES        //
    private float comulativeReward;
    private float currentReward;
    // METHODS //
    /* reset completely all reward system (every epoch) */
    public void ResetReward() 
    {
        comulativeReward = 0;
        currentReward = 0;
    }

    /* reset current reward (every step) */
    public void ZeroCurrentReward() 
    {
        comulativeReward += currentReward;
        currentReward = 0;
    }

    /* every step */
    public float GetReward() 
    {
        return currentReward;
    }

    /* for debug purpose */
    public float GetComulativeReward() 
    {
        return comulativeReward;
    }

    /* calculate speed-based reward */
    private void AddVelocityReward(float velocity, float maxSpeed, float direction)
    {
        float reward = stayStillPen;
        if (velocity >= stayStillLimit)
        {
            float normalizedSpeed = (velocity / maxSpeed) * direction;
            reward = normalizedSpeed * velocityRewardCostant;
        }
        currentReward += reward;
    }

    /* calculate speed-based reward */
    private void AddCooperativeVelocityReward(float velocity, float maxSpeed, float direction, float avgSpeed) 
    {
        float reward = stayStillPen;
        if (velocity >= stayStillLimit)
        {
            float normalizedSpeed = (velocity / maxSpeed) * direction;
            reward = (0.7f * normalizedSpeed + 0.3f * avgSpeed) * velocityRewardCostant;
        }
        currentReward += reward;
    }

    /* calculate location-based reward */
    private void AddCheckpointReward(int previousCheckpoint, int currentCheckpoint, int numberOfCheckpoints)
    {
        if (currentCheckpoint == (previousCheckpoint + 1) % numberOfCheckpoints)
        {
            currentReward += checkpointReward;
        }
        else if (currentCheckpoint == (previousCheckpoint - 1) % numberOfCheckpoints)
        {
            currentReward -= checkpointReward;
        }
    }

    /* calculate collision-based reward */
    private void AddCollisionReward(bool collision)
    {
        if(collision)
        {
            currentReward += collisionPen;
        }
    }

    /* collect and calculate current reward (once every step) */
    public void CollectReward(
        TRAIN_SETTING trainSetting,     //training setting (selfish, cooperative, competative)
        float velocity,                 // car velocity magnitude
        float maxSpeed,                 // max speed normalization constant
        float direction,                // direction of drive
        float avgSpeedOfOtherAgents,    // average speed of the rest of the cars.
        int previousCheckpoint,         // index of last checkpoint in previous step
        int currentCheckpoint,          // index of last checkpoint in current step
        int numberOfCheckpoints,        // number of checkpoints in race track
        bool collision                  // did a collision accur
        )
    {
        if (trainSetting == TRAIN_SETTING.COOPERATIVE)
        {
            AddCooperativeVelocityReward(velocity, maxSpeed, direction, avgSpeedOfOtherAgents);
        }
        else // Selfish or Competative
        {
            AddVelocityReward(velocity, maxSpeed, direction);
        }
        AddCheckpointReward(previousCheckpoint, currentCheckpoint, numberOfCheckpoints);
        AddCollisionReward(collision);
    }
}

public class MyRaceAgent : Agent
{
    SimpleCarControllers controller;        // car controller
    public GameObject firstCheckpoint;      // init param - reference to the next checkpoint on track
    private RayPerception rayPercept;       // sight detection component 
    private AgentManager manager;           // a way to get data of the other agents on track
    private TRAIN_SETTING trainSetting;     // selfish / competetive / cooperative setting.
    private AgentRewardSystem reward;       //reward system
    private const float maxSpeed = 6f;      // for speed normalization purposes, checked in editor.

    // STATE VARIABLES //
    private float direction;                // direction of drive (+1 if driving towards the next checkpoint, otherwise -1.)
    private float normalizedSpeed;          // last value of normalized speed (used only in cooperative mode)
    private Vector3 nextCheckpointPos;      // will use this location only to determine the +-1 direction (updated by checkpoint manager)
    private int previousCheckpointIdx;      // index of last checkpoint passed according to previous agent step
    private int currentCheckpointIdx;       // index of last checkpoint passed according to current agent step (updated by checkpoint manager)
    private bool gameOver;                  // a flag to signal game over (used in training when colliding with another car)
    private bool isCollision;               // a flag to signal if a collision accured (any collision)
    private float averageSpeed;             // average speed of all agents (used only in cooperative mode)


    /* reset all state variables */
    private void ResetAgent()
    {
        gameOver = false;
        isCollision = false;
        currentCheckpointIdx = -1;
        previousCheckpointIdx = -1;
        direction = 1;
        reward.ResetReward();
        normalizedSpeed = 0;
        nextCheckpointPos = firstCheckpoint.transform.position;
        averageSpeed = 0;
    }

    /* agent initialization */
    public override void InitializeAgent()
    {
        base.InitializeAgent();
        manager = GetComponentInParent<AgentManager>();
        trainSetting = manager.GetTrainSetting();
        rayPercept = GetComponent<RayPerception>();
        controller = gameObject.GetComponent<SimpleCarControllers>();
        reward = new AgentRewardSystem();
        ResetAgent();
    }

    /* update state - average speed and normalized speed (cooperative mode) */
    private void UpdateCooperativeAverageSpeed()
    {
        float velocity = controller.GetVelocity();
        float newNormalizedSpeed = (velocity / maxSpeed) * direction;
        float averageSpped = manager.UpdateAverageSpeed(normalizedSpeed, newNormalizedSpeed);
        normalizedSpeed = newNormalizedSpeed;
    }

    /* update state - direction */
    private void UpdateDirection()
    {
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
        float project = Vector3.Dot(velocity, nextCheckpointPos - gameObject.transform.position);
        direction = (project > 0) ? 1 : -1;
    }

    /* update state, get reward, and set next action */
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if(gameOver == true)
        {
            SetReward(-1f);
            Done();
        }
        
        /* set action */
        controller.SetInput(new Vector2(vectorAction[0], vectorAction[1]));

        /* collect reward based on previous state*/
        Debug.Log("whay did i get here..?");
        reward.CollectReward(trainSetting, controller.GetVelocity(), maxSpeed, direction,
            averageSpeed, previousCheckpointIdx, currentCheckpointIdx, manager.GetNumberOfCheckpoints(), isCollision);
        float reward_val = reward.GetReward();

        /* update internal state */
        UpdateCooperativeAverageSpeed();
        reward.ZeroCurrentReward();
        isCollision = false;
        previousCheckpointIdx = currentCheckpointIdx;
        UpdateDirection();
        Debug.Log("new reward: " + reward.GetComulativeReward());

        /* finish and set reward */
        if (IsDone() == false)
        {
            SetReward(reward_val);
        }
        if (reward.GetComulativeReward() < -10f)
        {
            Done();
        }
    }

    /* when collision accurs */
    private void OnCollisionEnter(Collision collision)
    {
        isCollision = true;
        if (manager.GetIsInference()==false && collision.transform.tag == "Player")
        {
            gameOver = true;
        }
    }

    /* observations */
    public override void CollectObservations()
    {
        AddVectorObs(direction);
        AddVectorObs(controller.GetVelocity() / maxSpeed);

        float rayDistance_long = 10f;
        float rayDistance_mid = 7.5f;
        float rayDistance_short = 5f;
        float[] rayAngles_long = {85f, 90f, 95f };
        float[] rayAngles_mid = { 60f, 75f, 105f, 120f};
        float[] rayAngles_short = { 0f, 45f, 135f, 180f };
        string[] detectableObjects = { "wall", "Player" };

        AddVectorObs(rayPercept.Perceive(rayDistance_long, rayAngles_long, detectableObjects, 0.1f, 0));
        AddVectorObs(rayPercept.Perceive(rayDistance_mid, rayAngles_mid, detectableObjects, 0.1f, 0));
        AddVectorObs(rayPercept.Perceive(rayDistance_short, rayAngles_short, detectableObjects, 0.1f, 0));
    }

    /* reset agent between episodes */
    public override void AgentReset()
    {
        gameObject.GetComponent<SimpleCarControllers>().Reset();
        ResetAgent();
    }

    /* called by checkpoint manager */
    public void CheckpointTrigger(Vector3 _nextCheckpointPos, int _currentCheckpointIdx)
    {
        currentCheckpointIdx = _currentCheckpointIdx;
        nextCheckpointPos = _nextCheckpointPos;
    }
}
