using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObstaclesManager : MonoBehaviour
{
    private void Start()
    {
       
    }
    List<GameObject> obstacles;
    private int NObstacles;
    private System.Random rnd = new System.Random();

    void ShuffleList()
    {
        for(int i=NObstacles-1; i>1; i--)
        {
            int k = rnd.Next(i);
            GameObject tmp = obstacles[k];
            obstacles[k] = obstacles[i];
            obstacles[i] = tmp;
        }
    }

    public void SetObstacles(int numberOfObstacles)
    {
        NObstacles = 0;
        obstacles = new List<GameObject>();
        foreach (Transform cp in transform)
        {
            if (cp.parent != null)
            {
                obstacles.Add(cp.gameObject);
                NObstacles++;
            }
        }
        for (int i = 0; i < NObstacles; i++)
        {
            obstacles[i].SetActive(true);
            Debug.Log("SetActive " + i);
        }

        ShuffleList();
        Debug.Log("Number of active obstacles: " + numberOfObstacles);
        for (int i = numberOfObstacles; i < NObstacles; i++)
        {
            obstacles[i].SetActive(false);
            Debug.Log("SetActive " + i);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
