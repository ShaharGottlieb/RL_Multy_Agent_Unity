using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Checkpoint : MonoBehaviour
{
    void OnTriggerEnter(Collider other)
    {
        MyRaceAgent agent = other.gameObject.GetComponent<MyRaceAgent>();
        CheckpointManager manager = GetComponentInParent<CheckpointManager>();
        manager.CheckpointTrigger(gameObject, agent);
    }
}


