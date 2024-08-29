#include "main.hpp"
#include "Dataset.hpp"
/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */
struct kDTreeNode
{
    vector<int> data;
    kDTreeNode *left;
    kDTreeNode *right;
    int label;
    kDTreeNode(vector<int> data, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {
        this->data = data;
        this->left = left;
        this->right = right;
    }
    kDTreeNode(int label, vector<int> data, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {   
        this->label = label;
        this->data = data;
        this->left = left;
        this->right = right;
    }
    friend ostream &operator<<(ostream &os, const kDTreeNode &node)
    {
        os << "(";
        for (int i = 0; i < node.data.size(); i++)
        {
            os << node.data[i];
            if (i != node.data.size() - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};
struct Point{
    int label;
    vector<int> data;
    Point(){}
    Point(int label, vector<int>data) {
        this->label = label;
        this->data = data;
    }
    const Point& operator=(const Point& other){
        this->label = other.label;
        this->data = other.data;
        return *this;
    }
};

class kDTree
{
private:
    int k;
    kDTreeNode *root;

    kDTreeNode * copy(kDTreeNode * root);
    void clear(kDTreeNode * root);
    void inorderTraversalRec(kDTreeNode * root) const;
    void preorderTraversalRec(kDTreeNode * root) const;
    void postorderTraversalRec(kDTreeNode * root) const;
    int heightRec(kDTreeNode * root) const;
    int leafCountRec(kDTreeNode * root) const;
    void insertRec(const vector<int> &point, kDTreeNode * &root, int level);
    bool vectorCompare(const vector<int> & v1, const vector<int> &v2);
    kDTreeNode * findMinNode(kDTreeNode * root, int alpha);
    void removeRec(const vector<int> &point,kDTreeNode * &root, int level);
    bool searchRec(const vector<int> &point, kDTreeNode * root, int level);
    void merge( vector<vector<int>> &pointList, int start, int mid, int end, int alpha);
    void mergeSort( vector<vector<int>> &pointList, int start, int end, int alpha);
    void buildTreeRec(vector<vector<int>> &pointList,int start, int end, kDTreeNode * &root, int level);
    double distanceEuclidean(const vector<int> &target, const vector<int> &point);
    void nearestNeighbourRec(const vector<int> &target, kDTreeNode * &best, kDTreeNode * root, int level);
    void bestListInsert(kDTreeNode * root, double distance, list<pair<kDTreeNode *, double>> &bestList);
    void kNearestNeighbourRec(const vector<int> &target, int k, list<pair<kDTreeNode *, double>> &bestList, kDTreeNode * root, int level);
public:
    kDTree(int k = 2);
    ~kDTree();
    kDTree(const kDTree &other);
    const kDTree &operator=(const kDTree &other);
    void inorderTraversal() const;
    void preorderTraversal() const;
    void postorderTraversal() const;
    int height() const;
    int nodeCountRec(kDTreeNode * root) const;
    int nodeCount() const;
    int leafCount() const;
    void insert(const vector<int> &point);
    void remove(const vector<int> &point);
    bool search(const vector<int> &point);
    void buildTree(const vector<vector<int>> &pointList);
    void nearestNeighbour(const vector<int> &target, kDTreeNode * &best);
    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList);

    void mergeLB( vector<Point> &pointList, int start, int mid, int end, int alpha);
    void mergeSortLB( vector<Point> &pointList, int start, int end, int alpha);
    void buildTreeLBRec(vector<Point> &pointList,int start, int end, kDTreeNode * &root, int level);
    void buildTreeLB(vector<Point> pointList);
};

class kNN
{
private:
    int k;
    Dataset *X_train;
    Dataset *y_train;

public:
    kNN(int k = 5);
    void fit(Dataset &X_train, Dataset &y_train);
    Dataset predict(Dataset &X_test);
    double score(const Dataset &y_test, const Dataset &y_pred);
};

// Please add more or modify as needed
