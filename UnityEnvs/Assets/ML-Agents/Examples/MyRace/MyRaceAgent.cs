using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class MyRaceAgent : Agent
{
    //public GameObject car;
    private Vector3 carStartPos;

    private void Start()
    {
        carStartPos = gameObject.transform.position;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {

        SimpleCarControllers controller = gameObject.GetComponent<SimpleCarControllers>();
        controller.SetInput(new Vector2(vectorAction[0],vectorAction[1]));
/*
        if (IsDone() == false)
        {
            float dist_x = ball.transform.position.x - gameObject.transform.position.x;
            float dist_z = ball.transform.position.z - gameObject.transform.position.z;
            float distFromCenter = dist_x * dist_x + dist_z * dist_z;
            SetReward(1 / (1 + distFromCenter));
        }
        
        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x ) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f )
        {
            Done();
            SetReward(-1f);
        }
        */
    }

    public override void CollectObservations()
    {
        /*
        AddVectorObs(gameObject.transform.rotation.z);
        AddVectorObs(gameObject.transform.rotation.x);
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity.x);
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity.y);
        AddVectorObs(ball.transform.GetComponent<Rigidbody>().velocity.z);
        AddVectorObs(ball.transform.position.x - gameObject.transform.position.x);
        AddVectorObs(ball.transform.position.y - gameObject.transform.position.y);
        AddVectorObs(ball.transform.position.z - gameObject.transform.position.z);
        */
    }

    public override void AgentReset()
    {
        gameObject.transform.position = carStartPos;
    }
}
