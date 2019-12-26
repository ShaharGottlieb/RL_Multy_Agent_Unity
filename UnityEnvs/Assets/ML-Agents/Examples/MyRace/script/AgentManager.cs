using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentManager : MonoBehaviour
{
    private List<MyRaceAgent> agents;
    private int NAgents;
    private TRAIN_SETTING maSetting;    //0 - defualt (selfish)
                                        // 1 - cooperative
                                        // 2 - competative
    private float averageSpeed;
    private bool isInference;
    private int numCheckpoints;
    // Start is called before the first frame update
    public void SetAgents(int numberOfAgents, TRAIN_SETTING setting, bool isInfer)
    {
        isInference = isInfer;
        maSetting = setting;
        NAgents = 0;
        agents = new List<MyRaceAgent>();
        foreach (Transform cp in transform)
        {
            if (cp.parent != null)
            {
                agents.Add(cp.gameObject.GetComponent<MyRaceAgent>());
                NAgents++;
            }
        }
        Debug.Log("Number of active agents: " + numberOfAgents);
        for (int i = numberOfAgents; i < NAgents; i++)
        {
            agents[i].gameObject.SetActive(false);
        }
    }

    public bool GetIsInference()
    {
        return isInference;
    }

    public int GetNumberOfCheckpoints()
    {
        if (numCheckpoints == -1)
        {
            numCheckpoints = GameObject.Find("Checkpoints").transform.childCount;
        }
        return numCheckpoints;
    }

    public TRAIN_SETTING GetTrainSetting()
    {
        return maSetting;
    }

    public float UpdateAverageSpeed(float previousSpeed, float newSpeed)
    {
        averageSpeed += ((newSpeed - previousSpeed) / NAgents); //update the avarage speed
        return averageSpeed;
    }

    void Start()
    {
        numCheckpoints = -1;
        averageSpeed = 0;
    }
    // Update is called once per frame
    void Update()
    {

    }
}
