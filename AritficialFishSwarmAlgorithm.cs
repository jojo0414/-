using System.Collections;
using UnityEngine;

public class AritficialFishSwarmAlgorithm : MonoBehaviour
{
    public Vector3 minBorder = new Vector3(-14f, -11.5f, 140f); //���U�e
    public Vector3 maxBorder = new Vector3(15f, 2.5f, 183f);   //�k�W��

    private float visual;       //���Z(�P���Z��)
    private int try_num;        //���զ���
    private float delta;      //�����]�l(0~1)
    private float eps;      //�̤p�ؼШ�Ʈt����

    //private float v;
    //private Quaternion[] lookRotation;

    public struct Fish
    {
        public Vector3 currentPosition;
        public Vector3 tempPosition;
        public Vector3 nextPosition;
        //public float lastFoodDensity;
        public float currentFoodDensity;
        public float tempFoodDensity;   //�����ثe�����U�@�B�̰��������@��
        public int noChangeNum;     //�����ثe�����S���j�T���ܤƪ��ֿn����
        public int tryNum;

        public bool getNext;    //�����O�_�w�g����ت��a
        public float step;
    }

    public struct Center
    {
        public Vector3 localPosition;
        public Vector3 globalPosition;
    }

    private GameObject[] fishesObject;
    private GameObject[] foodsObject;
    private Fish[] fishes;
    private Center fishCenter;

    void GetCenter()
    {
        Vector3 totalLocal = Vector3.zero;
        Vector3 totalGlobal = Vector3.zero;
        for(int i = 0; i < fishes.Length; i++)
        {
            totalGlobal += fishesObject[i].transform.GetChild(3).transform.position;
            totalLocal += fishesObject[i].transform.GetChild(3).transform.localPosition;
        }
        fishCenter.globalPosition = totalGlobal / fishes.Length;
        fishCenter.localPosition = totalLocal / fishes.Length;
        return;
    }

    // Start is called before the first frame update
    void Start()
    {
        visual = 15f;
        try_num = 50;  
        delta = 0.0618f;
        eps = 0.1f;

        fishesObject = GameObject.FindGameObjectsWithTag("fish");

        fishes = new Fish[fishesObject.Length];
        for (int i = 0; i < fishes.Length; i++)
        {
            fishes[i].currentPosition = fishesObject[i].transform.GetChild(3).transform.position;
            fishes[i].nextPosition = fishes[i].currentPosition;
            fishes[i].tempPosition = fishes[i].currentPosition;
            fishes[i].noChangeNum = 0;
            fishes[i].tryNum = 0;
            fishes[i].getNext = true;
        }
    }

    // Update is called once per frame
    void Update()
    {
        foodsObject = GameObject.FindGameObjectsWithTag("food");

        for (int i = 0; i < fishes.Length; i++)
        {
            if (fishes[i].getNext == true)
            {
                StartCoroutine(AF_Leap(i));
                fishes[i].step = Random.Range(2f, 5f);
                fishes[i].getNext = false;
            }


            //�]���䤣�쬰�ƻ�w�C����V�|�X���A�N����۪�����ǥؼЪ�����(lookat)
            fishesObject[i].transform.LookAt(fishes[i].nextPosition);

            if (Vector3.Distance(fishes[i].currentPosition, fishes[i].nextPosition) > fishes[i].step)
            {
                fishesObject[i].transform.position += fishesObject[i].transform.forward * Time.deltaTime*fishes[i].step;
                fishes[i].currentPosition = fishesObject[i].transform.GetChild(3).transform.position;
                GetFoodDensity(i, false);
                if(fishes[i].currentFoodDensity > -0.001f)
                {
                    Debug.Log(string.Format("fish{0} eat food", i));
                }
                GetCenter();
            }
            else
            {
                fishes[i].getNext = true;
            }
        }
    }

    void GetFoodDensity(int fishNo, bool isTemp)
    {
        //�p�⭹���@��
        //�Z���C�[1�����@��-0.001
        float minDistance = 999;
        for(int i = 0; i< foodsObject.Length; i++)
        {
            float tempDistance;
            if (isTemp == true)
            {
                tempDistance = Vector3.Distance(fishes[fishNo].tempPosition, foodsObject[i].transform.position);
            }
            else
            {
                tempDistance = Vector3.Distance(fishes[fishNo].currentPosition, foodsObject[i].transform.position);
            }
            if (tempDistance < minDistance)
                minDistance = tempDistance;
         }
        if(isTemp == true)
        {
            fishes[fishNo].tempFoodDensity = -minDistance;
        }
        else
        {
            fishes[fishNo].currentFoodDensity = -minDistance;
        }
    }

    private IEnumerator AF_Move(int i)
    {
        do
        {
            fishes[i].tempPosition = Random.insideUnitSphere * visual + fishes[i].currentPosition;
            yield return null;

        } while (fishes[i].tempPosition.x < minBorder.x || fishes[i].tempPosition.x > maxBorder.x ||
                    fishes[i].tempPosition.y < minBorder.y || fishes[i].tempPosition.y > maxBorder.y ||
                    fishes[i].tempPosition.z < minBorder.z || fishes[i].tempPosition.z > maxBorder.z);
        yield break;
    }

    private IEnumerator AF_Prey(int i)
    {
        fishes[i].tryNum = 0;
        //�p�G�������զ��Ƥj��try_num�������X
        do
        {
            //�H�����������@�I�A�ýT�{���I�b���ڬ���
            yield return StartCoroutine(AF_Move(i));

            //�Y�H���I�������@�פj��ثe��m�A�h���H���I�e�i
            GetFoodDensity(i, true);
            if (fishes[i].currentFoodDensity < fishes[i].tempFoodDensity)
            {
                Debug.Log(string.Format("fish{0}: AF_Prey", i));
                fishes[i].nextPosition = fishes[i].currentPosition + ((fishes[i].tempPosition - fishes[i].currentPosition) / Vector3.Distance(fishes[i].tempPosition, fishes[i].currentPosition)) * Random.Range(0.1f, 1f);
                yield break;
            }

            fishes[i].tryNum++;
        } while (fishes[i].tryNum <= try_num && foodsObject.Length > 0);

        //�Y���զ��Ƥj��try_num�h�������X
        Debug.Log(string.Format("fish{0}: AF_Move", i));
        fishes[i].nextPosition = fishes[i].tempPosition;
        yield break;
    }

    private IEnumerator AF_Follow(int i)
    {
        //�p����񪺳����ƶq
        Collider[] cols = Physics.OverlapSphere(fishes[i].currentPosition, visual);
        for (int j = 0; j < cols.Length; j++)
        {
            if (cols[j].tag.Equals("Fish"))
            {
                fishes[i].tempPosition = cols[j].transform.GetChild(3).transform.position;
                GetFoodDensity(i, true);
                if (fishes[i].tempFoodDensity > fishes[i].currentFoodDensity)
                {
                    Debug.Log(string.Format("fish{0}: AF_Follow", i));
                    fishes[i].nextPosition = fishes[i].currentPosition + ((fishes[i].tempPosition - fishes[i].currentPosition) / Vector3.Distance(fishes[i].tempPosition, fishes[i].currentPosition)) * Random.Range(0.1f, 1f);
                    yield break;
                }
            }
        }
        yield return StartCoroutine(AF_Prey(i));
    }

    private IEnumerator AF_Swarm(int i)
    {
        GetCenter();
        Debug.Log(string.Format("fish{0} center globalPosition = {1}", i,  fishCenter.globalPosition));

        //�`�����s�����I�����s�ƶq
        fishes[i].tempPosition = fishCenter.globalPosition;
        GetFoodDensity(i, true);
        if (fishes[i].tempFoodDensity > fishes[i].currentFoodDensity)
        {
            fishes[i].nextPosition = fishes[i].currentPosition + ((fishes[i].tempPosition - fishes[i].currentPosition) / Vector3.Distance(fishes[i].tempPosition, fishes[i].currentPosition)) * Random.Range(0.1f, 1f);
            Debug.Log(string.Format("fish{0}: AF_Swarm, nextPosition = {1}", i, fishes[i].nextPosition));
            yield break;
        }
        else
        {
            yield return StartCoroutine(AF_Follow(i));
        }
    }

    private IEnumerator AF_Leap(int i)
    {
        GetFoodDensity(i, false);

        if (fishes[i].noChangeNum > try_num / 2)
        {
            yield return StartCoroutine(AF_Move(i));
            fishes[i].nextPosition = fishes[i].tempPosition;
            fishes[i].noChangeNum = 0;
            //Debug.Log(string.Format("fish{0}: AF_Leap", i));
        }
        else
        {
            yield return StartCoroutine(AF_Swarm(i));
        }

        if (foodsObject.Length > 0)
        {
            if (Mathf.Abs(fishes[i].currentFoodDensity - fishes[i].tempFoodDensity) < eps)
                fishes[i].noChangeNum++;
            else
                fishes[i].noChangeNum = 0;
        }

        Debug.Log(string.Format("fish{0}: nextPosition = {1}", i, fishes[i].nextPosition));
        yield break;
    }
}