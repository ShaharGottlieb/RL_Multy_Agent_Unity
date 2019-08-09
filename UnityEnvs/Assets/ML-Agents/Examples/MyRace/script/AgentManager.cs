using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentManager : MonoBehaviour
{
    private List<GameObject> agents;
    private int NAgents;
    // Start is called before the first frame update
    public void SetAgents(int numberOfAgents)
    {
        NAgents = 0;
        agents = new List<GameObject>();
        foreach (Transform cp in transform)
        {
            if (cp.parent != null)
            {
                agents.Add(cp.gameObject);
                NAgents++;
            }
        }
        Debug.Log("Number of active agents: " + numberOfAgents);
        for (int i = numberOfAgents; i < NAgents; i++)
        {
            agents[i].SetActive(false);
        }
    }

    void Start()
    {

    }
    // Update is called once per frame
    void Update()
    {

    }
}
