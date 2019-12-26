using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public enum TRAIN_SETTING
{
    SELFISH = 0,
    COOPERATIVE,
    COMPETATIVE,
    N_TRAIN_SETTINGS
};

public class MyRaceAcademy : Academy
{
    public override void AcademyReset()
    {
#if UNITY_EDITOR
        Debug.unityLogger.logEnabled = true;
#else
        Debug.unityLogger.logEnabled = false;
#endif
        int activeAgents    = (int)resetParameters["num_agents"];
        int activeObstacles = (int)resetParameters["num_obstacles"];
        TRAIN_SETTING maSetting = (TRAIN_SETTING)resetParameters["setting"];
        GameObject.Find("AgentManager").GetComponent<AgentManager>().SetAgents(activeAgents, maSetting, this.GetIsInference());
        GameObject.Find("ObstaclesManager").GetComponent<ObstaclesManager>().SetObstacles(activeObstacles);
    }

    public override void AcademyStep()
    {

    }

}
