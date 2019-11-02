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

public enum TRACK_SIZE
{
    BIG = 0,
    SMALL
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
        TRACK_SIZE track = (TRACK_SIZE)resetParameters["track_size"];

        GameObject.Find("AgentManager").GetComponent<AgentManager>().SetAgents(activeAgents, maSetting, this.GetIsInference());
        GameObject.Find("ObstaclesManager").GetComponent<ObstaclesManager>().SetObstacles(activeObstacles);

        GameObject.Find("AgentManagerSmall").GetComponent<AgentManager>().SetAgents(activeAgents, maSetting, this.GetIsInference());
        GameObject.Find("ObstaclesManagerSmall").GetComponent<ObstaclesManager>().SetObstacles(activeObstacles);
        if (track == TRACK_SIZE.SMALL)
        {
            GameObject.Find("BigTrack").SetActive(false);
        }
        else
        {
            GameObject.Find("SmallTrack").SetActive(false);
            GameObject.Find("CarCamera").GetComponent<Camera>().depth = 10;
        }
    }

    public override void AcademyStep()
    {

    }

}
