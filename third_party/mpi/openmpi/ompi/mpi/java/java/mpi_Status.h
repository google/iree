/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class mpi_Status */

#ifndef _Included_mpi_Status
#define _Included_mpi_Status
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     mpi_Status
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_mpi_Status_init
  (JNIEnv *, jclass);

/*
 * Class:     mpi_Status
 * Method:    getCount
 * Signature: (IIIIJJ)I
 */
JNIEXPORT jint JNICALL Java_mpi_Status_getCount
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong, jlong);

/*
 * Class:     mpi_Status
 * Method:    isCancelled
 * Signature: (IIIIJ)Z
 */
JNIEXPORT jboolean JNICALL Java_mpi_Status_isCancelled
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong);

/*
 * Class:     mpi_Status
 * Method:    getElements
 * Signature: (IIIIJJ)I
 */
JNIEXPORT jint JNICALL Java_mpi_Status_getElements
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong, jlong);

/*
 * Class:     mpi_Status
 * Method:    getElementsX
 * Signature: (IIIIJJ)Lmpi/Count;
 */
JNIEXPORT jobject JNICALL Java_mpi_Status_getElementsX
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong, jlong);

/*
 * Class:     mpi_Status
 * Method:    setElements
 * Signature: (IIIIJJI)I
 */
JNIEXPORT jint JNICALL Java_mpi_Status_setElements
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong, jlong, jint);

/*
 * Class:     mpi_Status
 * Method:    setElementsX
 * Signature: (IIIIJJJ)J
 */
JNIEXPORT jlong JNICALL Java_mpi_Status_setElementsX
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong, jlong, jlong);

/*
 * Class:     mpi_Status
 * Method:    setCancelled
 * Signature: (IIIIJI)V
 */
JNIEXPORT void JNICALL Java_mpi_Status_setCancelled
  (JNIEnv *, jobject, jint, jint, jint, jint, jlong, jint);

#ifdef __cplusplus
}
#endif
#endif
