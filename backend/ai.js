import express from "express";

const aiRouter = express.Router();

// Health check
aiRouter.get("/", (req, res) => {
  res.send({
    message: "✅ Server running",
    routes: {
      ask: "POST /ask",
    },
  });
});

// Proxy AI ask → FastAPI
aiRouter.post("/ask", async (req, res) => {  // ✅ FIXED PATH
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Missing question" });
    }

    // Call FastAPI
    const response = await fetch("http://0.0.0.0:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: question }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`FastAPI error: ${errText}`);
    }

    const data = await response.json();

    res.json({
      answer: data.answer,
      context: data.context,
    });
  } catch (err) {
    console.error("Error in /ai/ask proxy:", err);
    res.status(500).json({
      error: "Failed to fetch answer from AI service",
      details: err.message,
    });
  }
});

export default aiRouter;
