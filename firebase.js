// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyA-vOkfoZLkR8yZV66sjF0seNjlL8yGeZk",
  authDomain: "weather-appli-7d3e5.firebaseapp.com",
  projectId: "weather-appli-7d3e5",
  storageBucket: "weather-appli-7d3e5.appspot.com",
  messagingSenderId: "1073988031383",
  appId: "1:1073988031383:web:5169142b7c8257dea92ad2"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export { app };

