import React from "react";
import { Box, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";

const HomePage = () => {
    const navigate = useNavigate();

  return (
    <div className="home-container">
        <Box
        sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
        }}
        >
        <Button
            variant="contained"
            color="primary"
            onClick={() => navigate("/user")}
            sx={{ marginBottom: 2 }}
        >
            User
        </Button>
        <Button
            variant="contained"
            color="secondary"
            onClick={() => navigate("/admin")}
        >
            Admin
        </Button>
        </Box>
    </div>
  );
};

export default HomePage;