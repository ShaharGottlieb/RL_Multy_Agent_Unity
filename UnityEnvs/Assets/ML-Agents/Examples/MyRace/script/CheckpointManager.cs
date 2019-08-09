using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckpointManager : MonoBehaviour
{
    private List<GameObject> checkpoints;
    private int NCheckpoints;
    private MyRaceAcademy academy;
    // Start is called before the first frame update
    void Start()
    {
        NCheckpoints = 0;
        academy = GameObject.Find("Academy").GetComponent<MyRaceAcademy>();
        checkpoints = new List<GameObject>();
        foreach (Transform cp in transform)
        {
            if (cp.parent != null)
            {
                checkpoints.Add(cp.gameObject);
                NCheckpoints++;
            }
        }
        int activeCheckpoints = (int)academy.resetParameters["num_checkpoints"];
        for (int i = activeCheckpoints; i < NCheckpoints; i++)
        {
            checkpoints[i].SetActive(false);
        }
    }

    public void CheckpointTrigger(GameObject cp, MyRaceAgent agent)
    {
        int index = checkpoints.IndexOf(cp);
        Vector3 nxtCheckpointPos = checkpoints[(index + 1) % NCheckpoints].transform.position;
        agent.SetNextCheckpoint(nxtCheckpointPos);
        agent.SetNextReward(index, checkpoints.Count);
    }
}
