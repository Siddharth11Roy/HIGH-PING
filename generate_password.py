import streamlit_authenticator as stauth

passwords = ['admin123']
hashed_passwords = stauth.Hasher(passwords).generate()
print("Your hashed password is: highping123")
print(hashed_passwords[0])
