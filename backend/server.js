import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import rateLimit from "express-rate-limit";
import morgan from "morgan";

// Routers
import { FetchData } from "./fetch.js"; // Fetch Facebook posts
import aiRouter from "./ai.js";         // AI routes (ask + embeddings)

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

/* ------------------------
   Middleware
------------------------ */
app.use(cors({ origin: "*" })); // You can restrict origin in prod
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev")); // Logs HTTP requests

/* ------------------------
   Rate limiting (AI routes only)
------------------------ */
const aiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // 30 requests per IP per minute
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    res.status(429).json({ error: "Too many requests, slow down!" });
  },
});
app.use("/ai", aiLimiter);

/* ------------------------
   Routes
------------------------ */
// Fetch posts: GET /fetch/api/posts
app.use(FetchData);

// AI router: POST /ai/ask and POST /ai/generate-embeddings
app.use("/ai", aiRouter);

// Health check
app.get("/", (req, res) => {
  res.json({
    status: "✅ OK",
    message: "Server is running",
    endpoints: {
      fetchPosts: "/fetch/api/posts",
      ai: {
        ask: "/ai/ask",
        generateEmbeddings: "/ai/generate-embeddings",
      },
    },
  });
});

/* ------------------------
   404 handler
------------------------ */
app.use((req, res) => {
  res.status(404).json({ error: "Endpoint not found", path: req.originalUrl });
});

/* ------------------------
   Global error handler
------------------------ */
app.use((err, req, res, next) => {
  console.error("💥 Server error:", err);
  res.status(err.status || 500).json({
    error: "Something went wrong!",
    details: process.env.NODE_ENV === "production" ? undefined : err.message,
  });
});

/* ------------------------
   Start server
------------------------ */
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://0.0.0.0:${PORT}`);
});
