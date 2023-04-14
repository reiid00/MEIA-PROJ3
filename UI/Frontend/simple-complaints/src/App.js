import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HomePage from "./HomePage";
import AppUser from "./AppUser";
import AdminPage from "./AdminPage";
import { TicketProvider } from "./TicketContext";
import "./App.css";

function App() {
  return (
    <TicketProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/user" element={<AppUser />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </Router>
    </TicketProvider>
  );
}

export default App;