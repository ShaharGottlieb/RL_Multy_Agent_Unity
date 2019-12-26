using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckpointManager : MonoBehaviour
{
    private List<GameObject> checkpoints;
    private int NCheckpoints;
    // Start is called before the first frame update
    void Start()
    {
        NCheckpoints = 0;
        checkpoints = new List<GameObject>();
        foreach (Transform cp in transform)
        {
            if (cp.parent != null)
            {
                checkpoints.Add(cp.gameObject);
                NCheckpoints++;
            }
        }
    }

    public void CheckpointTrigger(GameObject cp, MyRaceAgent agent)
    {
        int index = checkpoints.IndexOf(cp);
        Vector3 nxtCheckpointPos = checkpoints[(index + 1) % NCheckpoints].transform.position;
        agent.CheckpointTrigger(nxtCheckpointPos, index);
    }
}
