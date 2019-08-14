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

    private const float maxSpeed = 6f; // for speed normalization purposes, checked in editor.
    private Vector3 nextCheckpointPos; //will use this location only to determine the +-1 direction

    private float debug_reward;
    private float nextCheckpointReward;

    private void ResetAgent()
    {
        prevCheckpoint = -1;
        currCheckpoint = -1;
        nextCheckpointReward = 0;
        direction = 1;
        debug_reward = 0;
    }

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        ResetAgent();
        rayPercept = GetComponent<RayPerception>();
        nextCheckpointPos = firstCheckpoint.transform.position;
        controller = gameObject.GetComponent<SimpleCarControllers>();

    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        controller.SetInput(new Vector2(vectorAction[0], vectorAction[1]));

        float velocityReward = controller.GetVelocity()/maxSpeed * direction * 0.02f;
        float reward = velocityReward; //for velocity

        if (nextCheckpointReward != 0)
        {
            reward += nextCheckpointReward; //for checkpoint
            nextCheckpointReward = 0;
        }

        debug_reward += reward;
        //Debug.Log("new reward: " + debug_reward);


        if (IsDone() == false)
        {
            SetReward(reward);
        }

    }

    private void OnCollisionEnter(Collision collision)
    {
        //debug_reward -= 0.4f;
        SetReward(-0.4f);
    }

    private void OnCollisionStay(Collision collision)
    {
        //debug_reward -= 0.01f;
        SetReward(-0.01f);
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
        float[] rayAngles_short = { 0f, 180f };
        string[] detectableObjects = { "wall", "Player" };

        AddVectorObs(rayPercept.Perceive(rayDistance_long, rayAngles_long, detectableObjects, 0.2f, 0));
        AddVectorObs(rayPercept.Perceive(rayDistance_mid, rayAngles_mid, detectableObjects, 0.2f, 0));
        AddVectorObs(rayPercept.Perceive(rayDistance_short, rayAngles_short, detectableObjects, 0.2f, 0));

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
            nextCheckpointReward = 0.2f;
        }
        else if (currCheckpoint == (prevCheckpoint - 1) % numberOfCheckpoints)
        {
            nextCheckpointReward = -0.2f;
        }
        prevCheckpoint = currCheckpoint;
    }


    public void SetNextCheckpoint(Vector3 checkpointPos)
    {
        nextCheckpointPos = checkpointPos;
    }
}
