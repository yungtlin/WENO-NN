// Cpp: Detailed implementaion of the sparse matrix operations
//
// Author: Yung-Tien Lin (UCLA)
// Created: June, 1, 2021
// Updated: June, 8, 2021

////////////
// Header //
////////////
#include "SparseMatrix.h"

///////////
// Vec1D //
///////////
// Constructor
Vec1D::Vec1D(int_type size_in){
    size = size_in;
    value = (value_type*) malloc(sizeof(value_type)*size);
    
    for(int i = 0; i < size; i++){
        value[i] = 0;
    }
}


void Vec1D::size_check(int_type size_in){
    if (size != size_in){
        perror("Vec1D size not match\n");
        exit (EXIT_FAILURE);
    }
}

void Vec1D::print_value(){
    printf("Vec1D size: %li\n", size);
    for(int i = 0; i < size; i++){
        printf(SP_PRINT_FORMAT, value[i]);
        printf("\n");
    }
    printf("\n");
}

// Operations //
// Elementwise operations 
// Addition (C = A + B)
void Vec1D::add(const Vec1D& A, const Vec1D& B){
    // Check Dimension
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D addition size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] + B.value[i];
    }
}

// Subtraction (C = A - B)
void Vec1D::sub(const Vec1D& A, const Vec1D& B){
    // Check Dimension
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D addition size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] - B.value[i];
    }
}

// Multiplication (C = A * B)
void Vec1D::mul(const Vec1D& A, const Vec1D& B){
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D multiplication size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] * B.value[i];
    }
}

// Division (C = A / B)
void Vec1D::div(const Vec1D& A, const Vec1D& B){ 
    // Check Dimension
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D division size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] / B.value[i];
    }
}

// Directed operations
// Added by (C += A)
void Vec1D::addby(const Vec1D& A){
    // Check Dimension
    if (size != A.size){
        perror("Vec1D addition size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] += A.value[i];
    }
}

// Subtacted by (C -= A)
void Vec1D::subby(const Vec1D& A){
    // Check Dimension
    if (size != A.size){
        perror("Vec1D addition size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] -= A.value[i];
    }
}

// Constant operations
// Multiplication (C = A * b)
void Vec1D::mul(const Vec1D& A, const value_type& b){
    size_check(A.size);

    for(int j = 0; j < size; j++){
        this->value[j] = A.value[j] * b;
    }
}

// Division (C = A / b)
void Vec1D::div(const Vec1D& A, const value_type& b){
    size_check(A.size);

    for(int j = 0; j < size; j++){
        this->value[j] = A.value[j] / b;
    }
}

// Sparse matrix operations
void Vec1D::mul(const SP_Matrix& A, const Vec1D& b){
    size_check(A.__ny);
    if (b.size != A.__nx){
        perror("Vec1D multiplication size not match");
        exit (EXIT_FAILURE);
    }

    value_type sum = 0;
    ListNode* current;
    value_type u;
    int_type key;
    for(int j = 0; j < size; j++){
        sum = 0;
        current = A.__csr_matrix.__sparseTable[j].__first;

        if (current != NULL){
            for(current; current != NULL; current = current->__right){
                key = current->__key;
                u = current->__value;
                sum += u*b.value[key];
            }
        }
        value[j] = sum;
    }   
}

void Vec1D::mul(const Vec1D& b, const SP_Matrix& A){
    size_check(b.size);
    size_check(A.__nx);

    if (b.size != A.__ny){
        perror("Vec1D multiplication size not match");
        exit (EXIT_FAILURE);
    }

    value_type sum = 0;
    ListNode* current;
    value_type u;
    int_type key;
    for(int i = 0; i < size; i++){
        sum = 0;
        current = A.__csc_matrix.__sparseTable[i].__first;

        if (current != NULL){
            for(current; current != NULL; current = current->__right){
                key = current->__key;
                u = current->__value;
                sum += u*b.value[key];
            }
        }
        value[i] = sum;
    }   
}


// Operators //
// Assignment (c = b)
void Vec1D::operator=(const Vec1D& B){
    if (size == 0){
        size = B.size;
        value = (value_type*) malloc(sizeof(value_type)*size);
    } else if (size != B.size){
        perror("Vec1D assignment size not match");
    }

    for (int i = 0; i < size; i++){
        value[i] = B.value[i];
    }
}

// Elementwise operators
// c = a + b
Vec1D Vec1D::operator+(const Vec1D& B){
    int_type nx = size;
    Vec1D C(nx);
    C.add(*this, B);
    return C;
}

// c = a - b
Vec1D Vec1D::operator-(const Vec1D& B){
    int_type nx = size;
    Vec1D C(nx);
    C.sub(*this, B);
    return C;
}


// c = a * b
Vec1D Vec1D::operator*(const Vec1D& B){
    int_type nx = size;
    Vec1D C(nx);
    C.mul(*this, B);
    return C;
}

// c = a / b
Vec1D Vec1D::operator/(const Vec1D& b){
    Vec1D c(size);

    c.div(*this, b);

    return c;
}

// Constant operators
// C = A * b
Vec1D Vec1D::operator*(const value_type& b){
    Vec1D C(size);

    C.mul(*this, b);

    return C;
}

// C = A / b
Vec1D Vec1D::operator/(const value_type& b){
    Vec1D C(size);

    C.div(*this, b);

    return C;
}


// Sparse matrix operators
// c = b.T * A
Vec1D Vec1D::operator*(const SP_Matrix& A){
    Vec1D c(size);

    c.mul(*this, A);

    return c;
}

// Vec1D Related Operations
value_type dot(const Vec1D& A, const Vec1D& B){
    if (A.size != B.size){
        perror("Inner product size not match");
    }
    value_type sum = 0;
    for (int i = 0; i < A.size; i++){
        sum += A.value[i] * B.value[i];
    }
    return sum;
}

// Vec1D Related Operators
// (C = b * A)
Vec1D operator*(const value_type& b, const Vec1D& A){
    Vec1D C(A.size);

    C.mul(A, b);

    return C;
}


////////////////
// LinkedList //
////////////////
// assign or insert value and ascending ordered
void LinkedList::assign(int_type key, value_type value){
    if (value == LINKEDLIST_OMIT_VALUE){
        remove(key);
        return;
    }

    // empty condition
    if (__size == 0){
        ListNode *newNode = new ListNode(key, value, this->__first, this->__last);
        this->__size++;
        this->__first = newNode;
        this->__last = newNode;
        return;
    }

    // linear search
    ListNode* current = this->__first;
    for(current; current != NULL; current = current->__right){
        if (key == current->__key){
            current->__value = value; // simple assignmemt
            return;
        }
        if (key < current->__key){
            ListNode* new_left = current->__left;
            ListNode* new_right = current;

            ListNode *newNode = new ListNode(key, value, 
                new_left, new_right);
            this->__size++;
            
            if (new_left == NULL){
                this->__first = newNode;
            }
            else{
                current->__left->__right = newNode;
            }

            current->__left = newNode;
            return;
        }
    }

    // outside the list
    ListNode *newNode = new ListNode(key, value, __last, NULL);
    __size++;
    __last->__right = newNode;
    __last = newNode;
}

//
void LinkedList::remove(int_type key){
    ListNode* current = this->__first;
    for(current; current != NULL; current = current->__right){
        if (key == current->__key){
            ListNode* current_left = current->__left;
            ListNode* current_right = current->__right;

            
            if (current == this->__first){
                this->__first = current_right; 
            }
            else{
                current_left->__right = current_right;    
            }

            if (current == this->__last){
                this->__last = current_left;
            }
            else{
                current_right->__left = current_left;    
            }
            delete current;
            this->__size--;
        }
    }
}

//
void LinkedList::clear(){
    int_type original_size = __size;
    int_type key;
    for(int i = 0; i < original_size; i++){
        key = __first->__key;
        remove(key);
    }
}

void LinkedList::reset_iter(){
    this->__current = this->__first;
}

std::tuple<int_type, value_type> LinkedList::get_next(){
    if (__current == NULL){
        perror("LinkedList get_next outside the range\n");
        exit (EXIT_FAILURE);
    }
    auto tup = std::make_tuple(__current->__key, __current->__value);
    __current = __current->__right;
    return tup;
}

value_type LinkedList::get_value(int_type key){
    ListNode* current = this->__first;
    for(current; current != NULL; current = current->__right){
        if (key == current->__key){
            return current->__value;
        }
    }
    return LINKEDLIST_OMIT_VALUE;
}

void LinkedList::print(){
    ListNode* current = this->__first;
    for(current; current != NULL; current = current->__right){
        printf("%li %f\n", current->__key, current->__value);
    }
    printf("\n");
}

int_type LinkedList::get_size(){
    return __size;
}

// Operators // 
void LinkedList::operator=(const LinkedList& B){
    this->clear();

    int_type key;
    value_type value;

    int_type B_size = B.__size;
    ListNode* current = B.__first;
    
    for(int i = 0; i < B_size; i++){
        key = current->__key;
        value = current->__value;
        this->assign(key, value);
        current = current->__right;
    }
}


////////////////
// CSR_Matrix //
////////////////void input_check(int_type y, int_type x)
// Constructor
CSR_Matrix::CSR_Matrix(int_type ny, int_type nx){
    __ny = ny;
    __nx = nx;
    __sparseTable = new LinkedList[__ny];
}

// Destructor
CSR_Matrix::~CSR_Matrix(){
    delete [] __sparseTable;
}

// Functions //
void CSR_Matrix::input_check(int_type y, int_type x){
    if (y >= __ny || x >= __nx){
        perror("CSR Matrix outside range\n");
        exit (EXIT_FAILURE);
    }
}

void CSR_Matrix::size_check(int_type ny, int_type nx){
    if (ny != __ny || nx != __nx){
        perror("CSR Matrices size not match\n");
        exit (EXIT_FAILURE);
    }
}

std::tuple<int_type, int_type> CSR_Matrix::shape(){
    return std::make_tuple(__ny, __nx);
}


value_type CSR_Matrix::get_value(int_type y, int_type x){
    this->input_check(y, x);
    return __sparseTable[y].get_value(x);
}

void CSR_Matrix::assign(int_type y, int_type x, value_type u){
    this->input_check(y, x);
    __sparseTable[y].assign(x,  u);
}

void CSR_Matrix::clear(){
    for (int i = 0; i < __ny; i++){
        __sparseTable[i].clear();
    }
}

void CSR_Matrix::set_zeros(){
    this->clear();
}

void CSR_Matrix::set_eye(){
    this->clear();
    int_type n_min = (__nx < __ny) ? __nx : __ny;
    for (int i = 0; i < n_min; i++){
        this->assign(i, i, 1);   
    }
}

void CSR_Matrix::print_value(){
    printf("[");
    for (int j = 0; j < __ny; j++){
        printf("[");
        for (int i = 0; i < __nx; i++){
            printf(SP_PRINT_FORMAT, this->get_value(j, i));
        }
        printf("]\n");
    }
    printf("]\n");
}

// Operation
void CSR_Matrix::add(int_type y, int_type x, value_type u){
    this->input_check(y, x);
    value_type u2 = get_value(y, x);
    u2 = u + u2;
    assign(y, x, u2);
}


void CSR_Matrix::operator=(const CSR_Matrix& B){
    // If csr not specified
    if (__ny == 0 && __nx == 0){
        __ny = B.__ny;
        __nx = B.__nx;
        __sparseTable = new LinkedList[__ny];
    }
    else{
        size_check(B.__ny, B.__nx);
    }
    for (int j = 0; j < __ny; j++){
        this->__sparseTable[j] = B.__sparseTable[j];
    }
}


//////////////////
// SparseMatrix //
//////////////////

// Constructor
SP_Matrix::SP_Matrix(int_type ny, int_type nx){
    __ny = ny;
    __nx = nx;
    __csr_matrix = CSR_Matrix(ny, nx);
    __csc_matrix = CSR_Matrix(nx, ny);
}


SP_Matrix::SP_Matrix(){
    __ny = 0;
    __nx = 0;
    __csr_matrix = CSR_Matrix();
    __csc_matrix = CSR_Matrix();
}


// Destructor
SP_Matrix::~SP_Matrix(){
    //printf("SP_Matrixed Killed\n");
}

// Functions //
std::tuple<int_type, int_type> SP_Matrix::shape(){
    return std::make_tuple(__ny, __nx);
}

value_type SP_Matrix::get_value(int_type j, int_type i){
    return __csr_matrix.get_value(j, i);
}

void SP_Matrix::assign(int_type j, int_type i, value_type u){
   __csr_matrix.assign(j, i, u);
   __csc_matrix.assign(i, j, u);
}

void SP_Matrix::add(int_type j, int_type i, value_type u){
   __csr_matrix.add(j, i, u);
   __csc_matrix.add(i, j, u);
}


void SP_Matrix::set_row_zero(int_type y){
    for(int i = 0; i < __nx; i++){
        this->assign(y, i, LINKEDLIST_OMIT_VALUE);
    }
}


// check if csr == csc
void SP_Matrix::sparse_check(){
    int_type size;
    int_type key;
    value_type u_r, u_c;

    // csr check csc
    for(int j = 0; j < __ny; j++){
        // get size
        size = __csr_matrix.__sparseTable[j].get_size();
        // start row-wise iteration
        __csr_matrix.__sparseTable[j].reset_iter();
        for(int count=0; count < size; count++){
            std::tie(key, u_r) = __csr_matrix.__sparseTable[j].get_next();
            u_c = __csc_matrix.get_value(key, j);

            if (u_c != u_r){
                perror("SpaseMatrix CSR -> CSC Check Failed\n");
                exit (EXIT_FAILURE);
            }
        }
    }

    // csc check csr
    for(int i = 0; i < __nx; i++){
        // get size
        size = __csc_matrix.__sparseTable[i].get_size();
        // start row-wise iteration
        __csc_matrix.__sparseTable[i].reset_iter();
        for(int count=0; count < size; count++){
            std::tie(key, u_c) = __csc_matrix.__sparseTable[i].get_next();
            u_r = __csr_matrix.get_value(key, i);
            
            if (u_r != u_c){
                perror("SpaseMatrix CSC -> CSR Check Failed\n");
                exit (EXIT_FAILURE);
            }
        }
    }
    //printf("SparseMatrix Check Pass\n");
}

void SP_Matrix::size_check(int_type ny, int_type nx){    
    if (ny != __ny || nx != __nx){
        perror("Sparse Matrices size not match\n");
        exit (EXIT_FAILURE);
    }
}

void SP_Matrix::update_csc(){
    __csc_matrix.clear();

    int_type nx;
    int_type i;
    value_type u;
    for(int j = 0; j < __ny; j++){
        // get size
        nx = __csr_matrix.__sparseTable[j].get_size();
        // start row-wise iteration
        __csr_matrix.__sparseTable[j].reset_iter();
        for(int count=0; count < nx; count++){
            std::tie(i, u) = __csr_matrix.__sparseTable[j].get_next();
            __csc_matrix.assign(i, j, u);
        }
    }
    this->sparse_check();
}



void SP_Matrix::clear(){
    __csr_matrix.clear();
    __csc_matrix.clear();
}

void SP_Matrix::set_zeros(){
    this->clear();
}

void SP_Matrix::set_eye(){
    __csr_matrix.set_eye();
    __csc_matrix.set_eye();
}

void SP_Matrix::print_value(){
    __csr_matrix.print_value();
}

// Operation
void SP_Matrix::addby(const SP_Matrix &A){
    this->size_check(A.__ny, A.__nx);
    
    // update based on csr 
    int_type ny, nx, key;
    value_type u;
    ny = A.__ny;

    
    for(int j = 0; j < ny; j++){
        nx = A.__csr_matrix.__sparseTable[j].get_size();

        // start iteration
        ListNode* current;
        current = A.__csr_matrix.__sparseTable[j].__first;
        for(int count = 0; count < nx; count++){
            key = current->__key;
            u = current->__value;
            __csr_matrix.add(j, key, u);
            current = current->__right;
        }
    }
    
    // update csc
    this->update_csc();
}


void SP_Matrix::subby(const SP_Matrix &A){
    this->size_check(A.__ny, A.__nx);
    
    // update based on csr 
    int_type ny, nx, key;
    value_type u;
    ny = A.__ny;

    
    for(int j = 0; j < ny; j++){
        nx = A.__csr_matrix.__sparseTable[j].get_size();

        // start iteration
        ListNode* current;
        current = A.__csr_matrix.__sparseTable[j].__first;
        for(int count = 0; count < nx; count++){
            key = current->__key;
            u = current->__value;
            __csr_matrix.add(j, key, -u);
            current = current->__right;
        }
    }
    
    // update csc
    this->update_csc();
}

void SP_Matrix::mulby(const value_type a){
    int_type ny, nx, key;
    value_type u;



    ny = __ny;
    for(int j = 0; j < ny; j++){
        nx = __csr_matrix.__sparseTable[j].get_size();
        
        // start iteration
        __csr_matrix.__sparseTable[j].reset_iter();
        for(int count = 0; count < nx; count++){
            std::tie(key, u) = __csr_matrix.__sparseTable[j].get_next();
            __csr_matrix.assign(j, key, a*u);
        }
    }

    // update csc
    this->update_csc();
}


void SP_Matrix::mul(const SP_Matrix& A, const SP_Matrix& B){
    this->size_check(A.__ny, B.__nx);
    if (A.__nx != B.__ny){
        perror("Matmul A, B not match");
    }

    this->clear();

    value_type sum;
    ListNode *node_A, *node_B;

    int_type key_A, key_B, size_A;

    for(int j = 0; j < __ny; j++){
        for(int i = 0; i < __nx; i++){
            sum = 0;
            node_A = A.__csr_matrix.__sparseTable[j].__first;
            node_B = B.__csc_matrix.__sparseTable[i].__first;

            size_A = A.__csr_matrix.__sparseTable[j].__size;

            //
            if ((node_A != NULL) && (node_B != NULL)){
                for(int i = 0; i < size_A; i++){
                    key_A = node_A->__key;

                    // break if key_B >= key_A
                    for(key_B = node_B->__key; key_B < key_A; key_B){
                        node_B = node_B->__right;
                        // hit end
                        if (node_B == NULL){
                            key_B = key_A + 1;
                            i = size_A;    
                        }
                        else{ // update
                            key_B = node_B->__key;    
                        }
                    }
                    // found match
                    if (key_A == key_B){

                        sum += (node_A->__value * node_B->__value);
                    }

                    node_A = node_A->__right;
                }
            }

            __csr_matrix.assign(j, i, sum);
        }
    }
    this->update_csc();
}


// Operator //
void SP_Matrix::operator=(const SP_Matrix& B){
    if (__ny == 0 && __nx == 0){
        __ny = B.__ny;
        __nx = B.__nx;        
    }else{
        this->size_check(B.__ny, B.__nx);    
    }
    
    __csr_matrix = B.__csr_matrix;
    __csc_matrix = B.__csc_matrix;

    this->sparse_check();
}

SP_Matrix SP_Matrix::operator+(const SP_Matrix& B){
    SP_Matrix C(__ny, __nx);
    C.size_check(B.__ny, B.__nx);
    C = *this;
    C.addby(B);

    return C;
}

SP_Matrix SP_Matrix::operator-(const SP_Matrix& B){
    SP_Matrix C(__ny, __nx);
    C.size_check(B.__ny, B.__nx);
    C = *this;
    C.subby(B);

    return C;
}

// C = A * B
SP_Matrix SP_Matrix::operator*(const SP_Matrix& B){
    SP_Matrix C(__ny, B.__nx);
    C.mul(*this, B);
    return C;   
}

// c = A * b
Vec1D SP_Matrix::operator*(const Vec1D& b){
    Vec1D c(__ny);
    c.mul(*this, b);
    return c;
}

// C = A * a
SP_Matrix SP_Matrix::operator*(const value_type& a){
    SP_Matrix C(__ny, __nx);
    C = *this;

    C.mulby(a);

    return C;
}







