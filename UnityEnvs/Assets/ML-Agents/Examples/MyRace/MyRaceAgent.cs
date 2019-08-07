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
    private int prevCheckpoint;
    private int currCheckpoint;
    private float timePenalty;
    private float direction;

    private Vector3 nextCheckpointPos; //will use this location only to determine the +-1 direction

    private float debug_reward;
    private float nextCheckpointReward;

    private void ResetAgent()
    {
        prevCheckpoint = -1;
        currCheckpoint = -1;
        nextCheckpointReward = 0;
        direction = 1;
    }

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        ResetAgent();
        rayPercept = GetComponent<RayPerception>();
        nextCheckpointPos = firstCheckpoint.transform.position;
        controller = gameObject.GetComponent<SimpleCarControllers>();

        debug_reward = 0;
        Debug.Log("start debugging: reward " + debug_reward);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        controller.SetInput(new Vector2(vectorAction[0], vectorAction[1]));

        //timePenalty += 0.0001f;
        //float reward = 0.1f-timePenalty; //for time

        float velocityReward = controller.GetVelocity() * direction * 0.1f;
        float reward = velocityReward; //for velocity

        Debug.Log("direction is: " + direction);
        Debug.Log("velocity reward is: " + velocityReward);
        Debug.Log("component is: " + gameObject.name);

        if (nextCheckpointReward != 0)
        {
            reward += nextCheckpointReward; //for checkpoint
            nextCheckpointReward = 0;
        }

        debug_reward += reward;
        Debug.Log("new reward: " + debug_reward);


        if (IsDone() == false)
        {
            SetReward(reward);
        }

        if(timePenalty > 10f)
        {
            Done();
        }

    }

    private void OnCollisionEnter(Collision collision)
    {
        debug_reward += -15f;
        SetReward(-10f);
    }

    private void OnCollisionStay(Collision collision)
    {
        debug_reward += -0.2f;
        SetReward(-0.1f);
    }

    public override void CollectObservations()
    {
        Vector3 velocity = GetComponent<Rigidbody>().velocity;
        float project = Vector3.Dot(velocity, nextCheckpointPos - gameObject.transform.position);
        direction = (project > 0) ? 1 : -1;
        AddVectorObs(direction);
        AddVectorObs(velocity);
        float rayDistance = 50f;
        float[] rayAngles = { 0f, 60f, 75f, 85f, 90f, 95f, 105f, 120f, 180f };
        string[] detectableObjects = { "wall" };

        AddVectorObs(rayPercept.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));

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
            nextCheckpointReward = 20;
            timePenalty = 0;
        }
        else if (currCheckpoint == (prevCheckpoint - 1) % numberOfCheckpoints)
        {
            nextCheckpointReward = -20;
        }
        prevCheckpoint = currCheckpoint;
    }


    public void SetNextCheckpoint(Vector3 checkpointPos)
    {
        nextCheckpointPos = checkpointPos;
    }
}
