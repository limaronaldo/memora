/**
 * GET /api/r2/* - Proxy images from R2 storage
 * Supports ?db=memora or ?db=ob1 parameter to select bucket
 *
 * Handles paths like:
 *   /api/r2/memora/images/123/0.jpg
 *   /api/r2/images/123/0.jpg?db=ob1
 */

interface Env {
  R2_MEMORA: R2Bucket;
  R2_OB1: R2Bucket;
  DEFAULT_DB?: string;
}

function getBucket(env: Env, dbName: string | null): R2Bucket {
  const name = dbName || env.DEFAULT_DB || "memora";
  if (name === "ob1") return env.R2_OB1;
  return env.R2_MEMORA;
}

export const onRequestGet: PagesFunction<Env> = async ({ env, params, request }) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");

  // params.path is an array of path segments
  const pathSegments = params.path as string[];
  if (!pathSegments || pathSegments.length === 0) {
    return new Response("Not found", { status: 404 });
  }

  // Join path segments
  let objectKey = pathSegments.join("/");

  // Handle both bucket-prefixed and non-prefixed paths
  // R2 stores files like: images/123/0.jpg
  // URLs might come as: memora/images/123/0.jpg or images/123/0.jpg
  let bucket: R2Bucket;
  if (objectKey.startsWith("memora/")) {
    objectKey = objectKey.slice(7); // Remove "memora/" prefix
    bucket = env.R2_MEMORA;
  } else if (objectKey.startsWith("ob1/")) {
    objectKey = objectKey.slice(4); // Remove "ob1/" prefix
    bucket = env.R2_OB1;
  } else {
    bucket = getBucket(env, dbName);
  }

  try {
    const object = await bucket.get(objectKey);

    if (!object) {
      return new Response("Not found", { status: 404 });
    }

    // Determine content type
    const contentType = object.httpMetadata?.contentType ||
      getContentType(objectKey) ||
      "application/octet-stream";

    const headers = new Headers();
    headers.set("Content-Type", contentType);
    headers.set("Cache-Control", "public, max-age=86400"); // 24 hour cache
    headers.set("Access-Control-Allow-Origin", "*");

    // Add ETag for caching
    if (object.httpEtag) {
      headers.set("ETag", object.httpEtag);
    }

    return new Response(object.body, { headers });
  } catch (error) {
    console.error("R2 fetch error:", error);
    return new Response("Internal server error", { status: 500 });
  }
};

function getContentType(path: string): string | null {
  const ext = path.split(".").pop()?.toLowerCase();
  const mimeTypes: Record<string, string> = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "ico": "image/x-icon",
    "bmp": "image/bmp",
  };
  return ext ? mimeTypes[ext] || null : null;
}
