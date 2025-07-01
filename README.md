So kannst du die Application auf der BW-Cloud deployen:

1. **Zugriff bekommen**
   * Terminal öffnen (SSH-Key in NextCloud): 
   * ```bash
     ssh -i AI_StudyApp.key debian@193.196.54.243
     ```
  * Kontrolliere, ob die IP noch stimmt

2. **REPO clonen:**
   ```bash
     git clone https://github.com:KevinBasener/AI_StudyApp_API.git
     git clone https://github.com/KevinBasener/AI_StudyApp.git
   ```

3. **Installiere Docker und Docker Compose**  
   * Hier ist erklärt wie man Docker installiert: https://docs.docker.com/engine/install/debian/
   * Hier ist erklärt wie man Docker-Compose installiert: https://docs.docker.com/compose/install/linux/

4. **Nenne die `docker-compose--fullstack.yaml` in `docker-compose.yaml` um**  

5. **Builden**  
   Verwende den folgenden Befehl, um die Container zu builden:
   ```bash
   docker-compose up -d
   ```

6. **Überprüfen Sie die laufenden Container**  
   Verwenden Sie den Befehl:
   ```bash
   docker ps -a
   ```
## Hinweise:
* Der SSH-Key für den Server ist in NextCloud sowie die Logindaten für die BW-Cloud.
* Wenn du Nutzer hinzufügen willst, nutze die Django Funktionen innerhalb des web_project_container: 
  ```bash
  docker exec -it <container_id> bash
  python manage.py create_superuser('name', 'password')
  ```
  Um in den Container reinzukommen nutze:
  ```bash
  docker exec -it <container_id> bash
  ```
* Wenn keine Datenbankschemas in der Postgresdatenbank sind, dann füge sie noch mithilfe des `create_tables.sql`-Skript hinzu innerhalb des DB-Container (Command oben um in die Konsole des Containers zu kommen):
* ```bash
  docker exec -i <container_id> psql -U user -d chatbot_db < create_table.sql
  ```
* Die Docker-Compose kann auch lokal laufen gelassen werden.
* Eine Deployment Pipeline wäre wahrscheinlich sinnvoll mittlerweile :)
* Für die Dokumentation der API kannst du den Swagger-UI nutzen, der unter `http://193.196.54.243:8000/docs` erreichbar (oder `localhost:8000/docs`, wenn dus lokal laufen lässt) ist.
* Bei Fragen, Problemen und Verzweiflung einfach mir schreiben: `kevin.basener@gmail.com`