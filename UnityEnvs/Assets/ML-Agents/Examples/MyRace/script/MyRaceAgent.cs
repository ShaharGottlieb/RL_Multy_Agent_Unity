using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class MyRaceAgent : Agent
{
    SimpleCarControllers controller;
    public GameObject firstCheckpoint;
    public int trackLength;
    private RayPerception rayPercept;
    private AgentManager manager;
    private TRAIN_SETTING trainSetting;
    private int prevCheckpoint;
    private int currCheckpoint;
    private float direction;
    private float normalizedSpeed;
    private Vector3 nextCheckpointPos; //will use this location only to determine the +-1 direction


    private const float maxSpeed = 6f; // for speed normalization purposes, checked in editor.
    private const float velocityRewardCostant = 0.05f;
    private const float collisionPen = -0.4f;
    private const float checkpointReward = 0.2f;

    private float nextCheckpointReward;
    private float nextCollisionReward;
    private float comulativeReward;

    private void ResetAgent()
    {
        prevCheckpoint = -1;
        currCheckpoint = -1;
        nextCheckpointReward = 0;
        nextCollisionReward = 0;
        direction = 1;
        comulativeReward = 0;
    }

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        manager = GetComponentInParent<AgentManager>();
        trainSetting = manager.GetTrainSetting();
        ResetAgent();
        rayPercept = GetComponent<RayPerception>();
        nextCheckpointPos = firstCheckpoint.transform.position;
        controller = gameObject.GetComponent<SimpleCarControllers>();
    }

    private float CalculateVelocityReward()
    {
        float reward;
        float velocityReward;

        float velocity = controller.GetVelocity();
        float newNormalizedSpeed = (velocity / maxSpeed) * direction;
        float avgSpeed = manager.UpdateAverageSpeed(normalizedSpeed, newNormalizedSpeed);
        normalizedSpeed = newNormalizedSpeed;

        switch (trainSetting)
        {
            case TRAIN_SETTING.COOPERATIVE:
                velocityReward = (0.7f*normalizedSpeed + 0.3f*avgSpeed) * velocityRewardCostant;
                break;
            case TRAIN_SETTING.COMPETATIVE:
            case TRAIN_SETTING.SELFISH:
            default:
                velocityReward = normalizedSpeed * velocityRewardCostant;
                break;
        }

        reward = (velocity <= 0.3f) ? -0.01f : velocityReward; //for velocity
        return reward;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        controller.SetInput(new Vector2(vectorAction[0], vectorAction[1]));

        float reward = CalculateVelocityReward();

        if (nextCheckpointReward != 0)
        {
            reward += nextCheckpointReward; //for checkpoint
            nextCheckpointReward = 0;
        }
        reward += nextCollisionReward;
        nextCollisionReward = 0;

        comulativeReward += reward;
        Debug.Log("new reward: " + comulativeReward);

        if (IsDone() == false)
        {
            SetReward(reward);
        }

        if (comulativeReward < -20)
        {
            Done();
        }

    }

    private void OnCollisionEnter(Collision collision)
    {
        nextCollisionReward = collisionPen;
    }

    public override void CollectObservations()
    {
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
        float project = Vector3.Dot(velocity, nextCheckpointPos - gameObject.transform.position);
        direction = (project > 0) ? 1 : -1;
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

    public override void AgentReset()
    {
        gameObject.GetComponent<SimpleCarControllers>().Reset();
        ResetAgent();
    }

    public void SetNextReward(int current, int numberOfCheckpoints)
    {
        currCheckpoint = current;
        if (currCheckpoint == (prevCheckpoint + 1) % numberOfCheckpoints)
        {
            nextCheckpointReward = checkpointReward;
        }
        else if (currCheckpoint == (prevCheckpoint - 1) % numberOfCheckpoints)
        {
            nextCheckpointReward = -checkpointReward;
        }
        prevCheckpoint = currCheckpoint;
    }


    public void SetNextCheckpoint(Vector3 checkpointPos)
    {
        nextCheckpointPos = checkpointPos;
    }
}
