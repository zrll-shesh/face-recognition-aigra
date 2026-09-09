import logging
import numpy as np

logger = logging.getLogger("face_recognition")


class FaceEngine:
    def __init__(self, model_name="buffalo_l", detection_size=(640, 640), ctx_id=-1):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=tuple(detection_size))
        logger.info("FaceEngine ready with model=%s ctx_id=%s", model_name, ctx_id)

    def detect(self, image):
        return self.app.get(image)

    def get_largest_face(self, image):
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def get_embedding(self, image):
        face = self.get_largest_face(image)
        if face is None:
            return None, None
        return face.embedding, face

    @staticmethod
    def cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def similarity_to_confidence(similarity, low=0.0, high=0.75):
        clipped = max(low, min(similarity, high))
        return round(((clipped - low) / (high - low)) * 100, 2)

    def search(self, embedding, database, top_k=3):
        pairs = database.all_embeddings()
        if not pairs:
            return []

        best_per_employee = {}
        for eid, name, vec in pairs:
            sim = self.cosine_similarity(embedding, vec)
            if eid not in best_per_employee or sim > best_per_employee[eid][1]:
                best_per_employee[eid] = (name, sim)

        ranked = sorted(
            [(eid, name, sim) for eid, (name, sim) in best_per_employee.items()],
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:top_k]

    def enroll_from_images(self, images):
        embeddings = []
        faces = []
        for image in images:
            embedding, face = self.get_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
                faces.append(face)
        return embeddings, faces
