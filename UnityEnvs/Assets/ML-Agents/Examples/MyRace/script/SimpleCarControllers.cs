using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleCarControllers : MonoBehaviour
{

    private float m_horizontalInput;
    private float m_verticalInput;
    private float m_steeringAngle;

    private Vector2 m_input;

    public WheelCollider frontDriverW, frontPassengerW;
    public WheelCollider rearDriverW, rearPassengerW;
    public Transform frontDriverT, frontPassengerT;
    public Transform rearDriverT, rearPassengerT;
    public float maxSteerAngle = 15;
    public float motorForce = 50;

    private Vector3 startPos;
    private Quaternion startRot;
    private Vector3 startVel;

    private void Start()
    {
        startPos = transform.position;
        startRot = transform.rotation;
        startVel = gameObject.GetComponent<Rigidbody>().velocity;
    }

    public void GetInput()
    {
        //m_horizontalInput = Input.GetAxis("Horizontal");
        //m_verticalInput = Input.GetAxis("Vertical");
        m_horizontalInput = m_input[0];
        m_verticalInput = m_input[1];
    }

    public float GetVelocity()
    {
        float v = gameObject.GetComponent<Rigidbody>().velocity.magnitude;
        return (v > 0.01)? v : 0;
    }

    public void SetInput(Vector2 input)
    {
        m_input = input;
    }

    private void Steer()
    {
        m_steeringAngle = maxSteerAngle * m_horizontalInput;
        frontDriverW.steerAngle = m_steeringAngle;
        frontPassengerW.steerAngle = m_steeringAngle;
    }

    private void Accelerate()
    {
        frontDriverW.motorTorque = m_verticalInput * motorForce;
        frontPassengerW.motorTorque = m_verticalInput * motorForce;
    }

    private void UpdateWheelPoses()
    {
        UpdateWheelPose(frontDriverW, frontDriverT);
        UpdateWheelPose(frontPassengerW, frontPassengerT);
        UpdateWheelPose(rearDriverW, rearDriverT);
        UpdateWheelPose(rearPassengerW, rearPassengerT);
    }

    private void UpdateWheelPose(WheelCollider _collider, Transform _transform)
    {
        Vector3 _pos = _transform.position;
        Quaternion _quat = _transform.rotation;

        _collider.GetWorldPose(out _pos, out _quat);

        _transform.position = _pos;
        _transform.rotation = _quat;
    }

    private void FixedUpdate()
    {
        GetInput();
        Steer();
        Accelerate();
        UpdateWheelPoses();
    }

    public void Reset()
    {
        transform.SetPositionAndRotation(startPos,startRot);
        GetComponent<Rigidbody>().velocity = startVel;
    }

}
